from config.message_queue import get_connection,get_channel,close_connection  # Assicurati che questa funzione restituisca il client del database
from services.model_service import ModelService
from services.repository_service import RepositoryService
from services.training_params_service import TrainingParamsService,QualityLevel
from utils.frame_extractor import FrameExtractor
from models.model import Engine
from converters.ply_to_gsplat_converter import save_splat_file,process_ply_to_splat
from datetime import datetime
import subprocess
import os
import sys
import json
import requests
import shutil
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import tempfile
import zipfile
import re
from utils.opacity_converter import convert_taming_opacity

model_service = ModelService()
repository_service = RepositoryService()
frame_extractor = FrameExtractor()
training_params_service = TrainingParamsService()



# Cartella per i file di staging (zip delle fasi)
S3_STAGING_PREFIX = os.getenv('S3_STAGING_PREFIX', 'staging')

# Cartella per i deliverable finali
S3_DELIVERY_PREFIX = os.getenv('S3_DELIVERY_PREFIX', 'delivery')

WORKING_DIR = os.getenv("MODEL_WORKING_DIR") 
GAUSSIAN_SPLATTING_API_URL = "http://gaussian-splatting-api:8050"
COLMAP_API_URL = "http://colmap-converter-api:8060"

POINT_CLOUD_BUILDING_PHASE_ZIP_NAME = "point_cloud_building_phase.zip"
TRAINING_PHASE_ZIP_NAME = "training_phase.zip"

# Creiamo la mappa che associa ciascun valore a un oggetto con api-url
engine_map = {
    'INRIA': { 'api-url': 'http://gaussian-splatting-api:8100' },
    'MCMC': { 'api-url': 'http://3dgs-mcmc-api:8101' },
    'TAMING': { 'api-url': 'http://taming-3dgs-api:8102' },
}

# Assicurati che la cartella esista
os.makedirs(WORKING_DIR, exist_ok=True)

class QueueJobService:

    def __init__(self):
        """Inizializza la connessione a RabbitMQ"""
        self.connection = get_connection()
        self.channel = get_channel(self.connection)

    def fail(self,model_id: str,phase_str : str,error_message: str):
        print(error_message)
        model_service.fail_phase(model_id, phase_str,error_message)
        model_service.update_model_status(model_id, {"overall_status": "FAILED"})
        
    async def handle_frame_extraction(self, ch, method, model_id, data):
        """
        Fase 1: Download del video e creazione dei fotogrammi con preprocessing SAM2
        """
        model_service.start_phase(model_id, "frame_extraction")
        model = model_service.get_model_by_id(model_id)
        if not model:
            self.fail(model_id,"frame_extraction",f"Error: No model found for model_id {model_id}")
            return
                    
        video_s3_key = model.video_s3_key
        if not video_s3_key:
            self.fail(model_id,"frame_extraction",f"Error: No video_s3_key found for model_id {model_id}")
            return
                    
        model_dir = os.path.join(WORKING_DIR, f"{model_id}")
        os.makedirs(model_dir, exist_ok=True)

        # 🚀 RIMOZIONE TEMPFILE - Usa percorso fisso per sfruttare la cache
        local_video_path = os.path.join(model_dir, 'input_video.mp4')
        
        try:
            # Il repository_service.download() gestisce già la cache internamente!
            # ✅ Se in cache: copia dalla cache al percorso locale
            # ⬇️ Se non in cache: scarica da S3 e copia in cache
            repository_service.download(video_s3_key, local_video_path)
                        
            thumbnail_suffix = f"thumbnail.jpg"
            thumbnail_s3_key = f"{S3_DELIVERY_PREFIX}/{model_id}/{thumbnail_suffix}"
            
            print(f"📽️ Processing video from: {local_video_path}")
            
            # Verifica che il video sia stato scaricato correttamente
            if not os.path.exists(local_video_path):
                raise Exception(f"Video file not found after download: {local_video_path}")
            
            video_size = os.path.getsize(local_video_path)
            print(f"📊 Video size: {video_size / 1024 / 1024:.2f} MB")
            
            engine = model.training_config.get('engine') if model.training_config else None
            quality_level = model.training_config.get('quality_level') if model.training_config else None
            if not engine:
                self.fail(model_id,"training",f"Error: No engine found in model {model_id}")
                return

            print(f"🔍 DEBUG: engine = {engine}")
            print(f"🔍 DEBUG: quality_level = {quality_level}")
            print(f"🔍 DEBUG: training_params_service type = {type(training_params_service)}")
            print(f"🔍 DEBUG: Chiamando generate_params...")

            generated_params = training_params_service.generate_params(Engine(engine),QualityLevel(quality_level))

            print(f"🔍 DEBUG: generate_params completato!")
            print(f"🔍 DEBUG: generated_params = {generated_params}")
            # Calcola parametri ottimizzati
            target_width = frame_extractor.calculate_target_width(local_video_path,generated_params.final_params['resolution'])
            target_fps = frame_extractor.calculate_extraction_fps(local_video_path)
            print(f"Using FPS: {target_fps}")
            print(f"Resizing images to: {target_width}")

            with tempfile.TemporaryDirectory() as temp_sharp_output:
                # 🆕 SHARP FRAMES CLI - Comando corretto
                cmd = [
                    "sharp-frames",
                    local_video_path,
                    temp_sharp_output,
                    "--selection-method", "best-n",  # o "uniform"
                    "--min-buffer", "3",  # o "uniform"
                    "--num-frames" , str(250)
                ]
                # Aggiungi --width solo se necessario
                if target_width is not None:
                    cmd.extend(["--width", str(target_width)])

                print(f"🔧 Running: {' '.join(cmd)}")
            
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print(f"✅ Sharp-frames completed successfully")
                    print(f"stdout: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Sharp-frames failed with exit code {e.returncode}")
                    print(f"📝 stdout: {e.stdout}")
                    print(f"🔴 stderr: {e.stderr}")
                    
                    # Debug aggiuntivo
                    print(f"🔍 Video file exists: {os.path.exists(local_video_path)}")
                    print(f"🔍 Video file size: {os.path.getsize(local_video_path) if os.path.exists(local_video_path) else 'N/A'} bytes")
                    print(f"🔍 Output dir exists: {os.path.exists(temp_sharp_output)}")
                    print(f"🔍 Output dir writable: {os.access(temp_sharp_output, os.W_OK)}")
            
                    raise Exception(f"Sharp-frames error: {e.stderr}")
            
                print(f"✅ Sharp Frames output: {result.stdout}")
                if result.stderr:
                    print(f"⚠️ Sharp Frames warnings: {result.stderr}")       
            
                sharped_frames = sorted([
                    f for f in os.listdir(temp_sharp_output)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                print(f"📸 Frame generati da sharp-frames: {len(sharped_frames)}")
                print(f"📂 Esempi frame: {sharped_frames[:5]}")
                

                frames_output_folder = os.path.join(model_dir, 'input')
                os.makedirs(frames_output_folder, exist_ok=True)

                filtered_count = frame_extractor.filter_similar_frames(
                temp_sharp_output,
                frames_output_folder,
                similarity_threshold=0.80  # oppure 0.70 a seconda del tuo target
                )

                # Conta i frame estratti
                frame_files = []
                if os.path.exists(frames_output_folder):
                    for f in sorted(os.listdir(frames_output_folder)):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            frame_files.append(os.path.join(frames_output_folder, f))
                print(f"📊 Sharp frames extracted: {len(filtered_count)}")

                # Gestione thumbnail
                thumbnail_path = frame_files[0] if frame_files else None
                print(f"✅ Thumbnail local path: {thumbnail_path} and exists? " + str(os.path.exists(thumbnail_path) if thumbnail_path else False))
                
                
                if thumbnail_path and os.path.exists(thumbnail_path):
                    try:
                        repository_service.upload(thumbnail_path, thumbnail_s3_key)
                        print(f"✅ Thumbnail caricata su S3: {thumbnail_s3_key}")
                    except Exception as e:
                        print(f"❌ Errore durante l'upload della thumbnail su S3: {e}")
                        thumbnail_s3_key = None

                # 📤 UPLOAD CARTELLA INPUT (FRAMES) SU S3
                self.create_phase_zip_and_upload(model_id,model_dir,POINT_CLOUD_BUILDING_PHASE_ZIP_NAME,['input'])


                # ✅ COMPLETE FASE con metadata
                phase_metadata = {
                    "frame_count": len(filtered_count),
                    "video_local_path": local_video_path,  # Path del video per debug
                    "video_size_mb": video_size / 1024 / 1024 if 'video_size' in locals() else None,
                    "processing_params": {
                        "outlier_window_size": 10,
                        "outlier_sensitivity": 60,
                        "selection_method": "outlier-removal",
                        "fps": target_fps,
                        "width": target_width
                    }
                }

                model_service.complete_phase(model_id, "frame_extraction", 
                                        metadata=phase_metadata)
                
                # Aggiorna il modello con thumbnail
                if thumbnail_suffix:
                    model_service.update_model_status(model_id, {
                        "thumbnail_suffix": thumbnail_suffix  # ✅ A livello root del modello
                    })
                
                self.send_to_next_phase(model_id, "point_cloud_queue")
            
        except Exception as e:
            print(f"❌ Error in frame extraction: {e}")
            self.fail(model_id, "frame_extraction", f"Frame extraction failed: {e}")
            return
        
        finally:
            # 🧹 CLEANUP OPZIONALE - Rimuovi il video locale dopo il processing
            # (mantieni solo se serve per debug, altrimenti risparmia spazio disco)
            try:
                if os.path.exists(local_video_path):
                    os.remove(local_video_path)
                    print(f"🧹 Cleaned up local video file: {local_video_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not cleanup video file: {e}")

    async def handle_point_cloud_building(self, ch, method, model_id, data):
        """Fase 2: Creazione del punto nuvola tramite Gaussian Splatting"""
        model_service.start_phase(model_id, "point_cloud_building")
        model = model_service.get_model_by_id(model_id)
        if not model:
            self.fail(model_id,"point_cloud_building",f"Error: No model found for model_id {model_id}")
            return
        
        # 🆕 Salva backup delle immagini segmentate PRIMA di COLMAP
        model_dir = os.path.join(WORKING_DIR, f"{model_id}")
        input_dir = os.path.join(model_dir, 'input')

        if os.path.exists(input_dir) and os.listdir(input_dir):
            print(f"✅ Directory input già esistente per model_id {model_id}")
        else:
            print(f"📥 Scaricando directory input da S3 per model_id {model_id}")
            # Crea directory se non esiste
            os.makedirs(model_dir, exist_ok=True)

            # Ottieni il percorso S3 del ZIP della fase point cloud building
            point_cloud_zip_s3_key = f"{S3_STAGING_PREFIX}/{model.parent_model_id}/{POINT_CLOUD_BUILDING_PHASE_ZIP_NAME}"

            # Scarica ed estrai lo ZIP
            success = self.download_and_extract_phase_zip(point_cloud_zip_s3_key, model_dir)

            if not success:
                self.fail(model_id, "point_cloud_building", 
                         f"Failed to download/extract point cloud building ZIP from {point_cloud_zip_s3_key}")
                return
            
             # Verifica nuovamente che l'estrazione sia andata a buon fine
            if not os.path.exists(input_dir):
                self.fail(model_id, "point_cloud_building", 
                         f"Input directory is still invalid after ZIP extraction")
                return
           
        # 2. Conta frame di input per metadata
        frame_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        input_frame_count = len(frame_files)

        # Cartella sparse per la point cloud
        sparse_dir = os.path.join(model_dir, 'sparse')


        print(f"Generando la nuvola di punti per model_id {model_id}...")
            
        print(f"🔄 Avvio generazione point cloud per model_id {model_id}...")
        colmap_start_time = datetime.utcnow()
        
        # COLMAP API call
        convert_request = {"input_dir": model_dir}
        response = requests.post(COLMAP_API_URL + "/convert", json=convert_request)
        
        # ⏱️ FINE TIMING COLMAP
        colmap_end_time = datetime.utcnow()
        colmap_duration = colmap_end_time - colmap_start_time
        colmap_duration_seconds = colmap_duration.total_seconds()
            
        if response.status_code != 200:
            self.fail(model_id,"point_cloud_building",f" Colmap error: {response.text}")
            return
        
        # 5. Verifica che COLMAP abbia creato i file
        if not os.path.exists(sparse_dir) or not os.listdir(sparse_dir):
            self.fail(model_id,"point_cloud_building",f"Error: no cloud point created for model_id {model_id}")
            return   
         
        print(f"✅ Nuvola di punti generata con successo per model_id {model_id}")

        self.create_phase_zip_and_upload(model_id,model_dir,TRAINING_PHASE_ZIP_NAME,['sparse', 'images'])
        

        # 7. Aggiorna stato e passa alla fase successiva
        phase_metadata = {
            "input_frame_count": input_frame_count,
            "colmap_duration_seconds": round(colmap_duration_seconds, 2),
            "colmap_start_time": colmap_start_time.isoformat(),
            "colmap_end_time": colmap_end_time.isoformat(),
        }

        model_service.complete_phase(model_id, "point_cloud_building", 
                                   metadata=phase_metadata)
    
        self.send_to_next_phase(model_id, "model_training_queue")

    async def handle_training(self, ch, method, model_id, data):
        print(f"🚀 Start training model {model_id}")
        
        # ⏰ START FASE
        model_service.start_phase(model_id, "training")
        
        try:
            model = model_service.get_model_by_id(model_id)
            if not model:
                self.fail(model_id,"training",f"Error: No model found for model_id {model_id}")
                return
            
            model_dir = os.path.join(WORKING_DIR, f"{model_id}")
            image_dir = os.path.join(model_dir, "images")
            sparse_dir = os.path.join(model_dir, "sparse")
            
            if os.path.exists(image_dir) and os.listdir(image_dir) and os.path.exists(sparse_dir) and os.listdir(sparse_dir):
                print(f"✅ Directories images e sparse già esistenti per model_id {model_id}")
            else:
                print(f"📥 Scaricando directory input da S3 per model_id {model_id}")
                # Crea directory se non esiste
                os.makedirs(model_dir, exist_ok=True)

                # Ottieni il percorso S3 del ZIP della fase point cloud building
                training_zip_s3_key = f"{S3_STAGING_PREFIX}/{model.parent_model_id}/{TRAINING_PHASE_ZIP_NAME}"

                # Scarica ed estrai lo ZIP
                success = self.download_and_extract_phase_zip(training_zip_s3_key, model_dir)
                if not success:
                    self.fail(model_id, "training", 
                            f"Failed to download/extract training ZIP from {training_zip_s3_key}")
                    return

            
             # Verifica nuovamente che l'estrazione sia andata a buon fine
            if not os.path.exists(image_dir)  or not os.path.exists(sparse_dir):
                self.fail(model_id, "training", 
                         f"Images or sparse directories are still invalid after ZIP extraction")
                return
            
            
            engine = model.training_config.get('engine') if model.training_config else None
            quality_level = model.training_config.get('quality_level') if model.training_config else None
            if not engine:
                self.fail(model_id,"training",f"Error: No engine found in model {model_id}")
                return
            
            # 3. Prepara directory di output
            train_output_folder = os.path.join(model_dir, 'output')
            os.makedirs(train_output_folder, exist_ok=True)
            
            generated_params = training_params_service.generate_params(Engine(engine),QualityLevel(quality_level))
            # 4. Prepara richiesta di training
            train_request = {
                "input_dir": model_dir,
                "output_dir": train_output_folder,
                "params": generated_params.final_params,
            }
            
            # 5. Chiamata API per il training
            api_url = engine_map.get(engine, {}).get('api-url')
            if not api_url:
                self.fail(model_id,"training",f"Error: No api url found for engine {engine}")
                return
            
            print(f"🎯 Starting training with engine: {engine}")
            training_start_time = datetime.utcnow()
        
            # TRAIN API call
            response = requests.post(f"{api_url}/train", json=train_request)
            # ⏱️ FINE TIMING TRAIN
            training_end_time = datetime.utcnow()
            training_duration = training_end_time - training_start_time
            training_duration_seconds = training_duration.total_seconds()
            
            if response.status_code != 200:
                self.fail(model_id,"training",f"Error: Training failed status code {response.status_code}")
                return
            
            print(f"✅ Training completato con successo")
            
            # Prima dell'upload
            print(f"🔍 Output folder exists: {os.path.exists(train_output_folder)}")

            # ✅ COMPLETE FASE
            phase_metadata = {
            "training_duration_seconds": round(training_duration_seconds, 2),
            "training_start_time": training_start_time.isoformat(),
            "training_end_time": training_end_time.isoformat(),
            "training_parameters": {
                "engine": engine
            }
        }

            model_service.complete_phase(model_id, "training", 
                                   metadata=phase_metadata)
            # 8. Passa alla fase successiva
            self.send_to_next_phase(model_id, "upload_queue")
        
        except Exception as e:
            self.fail(model_id,"training",f"Training error: {e}")

    async def handle_model_upload(self, ch, method, model_id, data):
        """Fase 4: Caricamento del modello addestrato su S3"""

        # ⏰ START FASE
        model_service.start_phase(model_id, "upload")
        
        try:
            model = model_service.get_model_by_id(model_id)
            if not model:
                self.fail(model_id,"training",f"Error: No model found for model_id {model_id}")
                return
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Crea i percorsi come al solito, ma per gsplat_path utilizziamo la cartella temporanea
                model_dir = os.path.join(WORKING_DIR, f"{model_id}")
                
                # 1. Verifica se sparse directory esiste (output di COLMAP)
                output_dir = os.path.join(model_dir, 'output')
                if not os.path.exists(output_dir):
                    self.fail(model_id,"upload",f"No folder output found: {e}")
                    return


                ply_path = self.find_latest_iteration_folder(output_dir)
                cameras_file_path = os.path.join(output_dir, "cameras.json")
            
                engine = model.training_config.get('engine') if model.training_config else None

                # Usa la cartella temporanea per il file .splat
                gsplat_path = os.path.join(temp_dir, "point_cloud.splat")
                # Processa il file .ply e salva il .splat nella cartella temporanea
                save_splat_file(process_ply_to_splat(ply_path,engine == Engine.TAMING.value), gsplat_path)

                # Aggiungi il file cameras.json alla cartella temporanea
                shutil.copy(cameras_file_path, temp_dir)

                # Creazione del file ZIP che contiene il file .splat e il file cameras.json
                zip_filename = os.path.join(model_dir, "3d_model.zip")
                shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', temp_dir)

                # Inizio del caricamento del modello su S3
                print(f"Inizio del caricamento del modello {model_id} su S3...")
                zip_model_suffix = f"3d_model.zip"
                zip_model_s3_key = f"{S3_DELIVERY_PREFIX}/{model_id}/{zip_model_suffix}"

                try:
                    repository_service.upload(zip_filename, zip_model_s3_key)
                    print(f"✅ Il modello {model_id} è stato caricato con successo su S3 nella chiave {zip_model_s3_key}")
                except Exception as e:
                    self.fail(model_id,"upload",f"Errore durante l'upload su S3: {e}")
                    return
                
               
                # Passa al prossimo job (upload)
                model_service.complete_phase(model_id, "upload")

                model_service.update_model_status(model_id, {"zip_model_suffix": zip_model_suffix})
                print(f"Model {model_id} caricato su S3 con successo!")
                self.send_to_next_phase(model_id, "metrics_generation_queue")
                
        except FileNotFoundError as e:
            self.fail(model_id,"upload",f"Error: {e}")
        except NoCredentialsError:
            self.fail(model_id,"upload","Error: AWS credential not found.")
        except PartialCredentialsError:
            self.fail(model_id,"upload","Error: AWS credential partially not found")
        except Exception as e:
            self.fail(model_id,"upload",f"Fail to upload model: {e}")
        
    def find_latest_iteration_folder(self,output_dir):
        """Trova la cartella iteration_XXXXX con il numero più alto"""
        point_cloud_dir = os.path.join(output_dir, "point_cloud")
        
        if not os.path.exists(point_cloud_dir):
            return None
        
        # Pattern per cartelle iteration_NUMERO
        iteration_pattern = re.compile(r'^iteration_(\d+)$')
        max_iteration = -1
        latest_folder = None
        
        for folder_name in os.listdir(point_cloud_dir):
            match = iteration_pattern.match(folder_name)
            if match:
                iteration_num = int(match.group(1))
                if iteration_num > max_iteration:
                    max_iteration = iteration_num
                    latest_folder = folder_name
        
        if latest_folder:
            ply_path = os.path.join(point_cloud_dir, latest_folder, "point_cloud.ply")
            print(f"🎯 Latest iteration folder: {latest_folder}")
            print(f"📄 Expected PLY path: {ply_path}")
            
            if os.path.exists(ply_path):
                file_size = os.path.getsize(ply_path)
                print(f"✅ PLY file found: {file_size} bytes")
                return ply_path
            else:
                print(f"❌ PLY file doesn't exist at: {ply_path}")
                return None
        else:
            print(f"❌ No iteration folders found")
            return None
        
    async def handle_metrics_generation(self, ch, method, model_id, data):
        """Fase 5: Generazione delle metriche e salvataggio su Mongo"""
         # ⏰ START FASE
        model_service.start_phase(model_id, "metrics_evaluation")
        
        try:
            model = model_service.get_model_by_id(model_id)
            if not model:
                self.fail(model_id,"metrics_evaluation",f"Error: No model found for model_id {model_id}")
                return
            
            model_dir = os.path.join(WORKING_DIR, f"{model_id}")
                
            # 1. Verifica se sparse directory esiste (output del training)
            output_dir = os.path.join(model_dir, 'output')
            if not os.path.exists(output_dir):
                self.fail(model_id,"metrics_evaluation",f"No folder output found: {e}")
                return

            engine = model.training_config.get('engine') if model.training_config else 'INRIA'

            render_request = { "output_dir": output_dir}
            response = requests.post(engine_map.get(engine).get('api-url') + "/render", json=render_request)
            response.raise_for_status()  # Controlla che non ci siano errori nel render

            metrics_request = { "output_dir": output_dir}
            response = requests.post(engine_map.get(engine).get('api-url') + "/metrics", json=metrics_request)
            response.raise_for_status()  # Controlla che non ci siano errori nel render

            # Verifica che il file results.json esista
            results_json_path = os.path.join(output_dir, "results.json")
            if not os.path.exists(results_json_path):
                raise FileNotFoundError("Il file 'results.json' non è stato trovato.")
            # Leggi il contenuto di results.json
            with open(results_json_path, 'r') as f:
                results_data = json.load(f)

            # Strategia 2: Cerca chiave che inizia con "ours_"
            ours_keys = [key for key in results_data.keys() if key.startswith("ours_")]
            results = None
            if len(ours_keys) == 1:
                # Solo una chiave "ours_", usala
                key = ours_keys[0]
                print(f"✅ Found single ours key: {key}")
                results = self._extract_metrics_from_section(results_data[key])
            elif len(ours_keys) > 1:
                def extract_number(key):
                    try:
                        return int(key.split("_")[1])
                    except (IndexError, ValueError):
                        return 0
                latest_key = max(ours_keys, key=extract_number)
                print(f"✅ Found multiple ours keys, using latest: {latest_key}")
                results = self._extract_metrics_from_section(results_data[latest_key])
            else:
                # Nessuna chiave "ours_", cerca qualsiasi chiave con metriche
                for key, value in results_data.items():
                    if isinstance(value, dict) and any(metric in value for metric in ["SSIM", "PSNR", "LPIPS"]):
                        print(f"✅ Found metrics in fallback key: {key}")
                        results = self._extract_metrics_from_section(value)
                
                raise KeyError(f"Nessuna sezione con metriche trovata nel file results.json. Chiavi disponibili: {list(results_data.keys())}")
            
            model_service.complete_phase(model_id, "metrics_evaluation",overall_status="COMPLETED",metadata={"metrics": results})
            print(f"Model {model_id} caricato su S3 con successo!")
        except FileNotFoundError as e:
            self.fail(model_id,"metrics_evaluation",f"Error: {e}")
        except NoCredentialsError:
            self.fail(model_id,"metrics_evaluation","Error: AWS credential not found.")
        except PartialCredentialsError:
            self.fail(model_id,"metrics_evaluation","Error: AWS credential partially not found")
        except Exception as e:
            self.fail(model_id,"metrics_evaluation",f"Fail to load model: {e}")
            
    def _extract_metrics_from_section(self, metrics_section):
        """Estrai metriche da una sezione specifica"""
        return {
            "ssim": metrics_section.get("SSIM", None),
            "psnr": metrics_section.get("PSNR", None), 
            "lpips": metrics_section.get("LPIPS", None)
        }
    def create_phase_zip(self,model_dir, folders_to_include, zip_path):
        """
        Crea un ZIP della fase training contenente le cartelle COLMAP specificate.
        
        Args:
            model_dir: Directory del modello contenente le cartelle
            folders_to_include: Lista delle cartelle da includere nel ZIP
            zip_path: Percorso dove creare il file ZIP
            
        Returns:
            bool: True se il ZIP è stato creato con successo
        """
        try:
            print(f"📦 Creating training phase ZIP: {zip_path}")
            print(f"📁 Folders to include: {folders_to_include}")
            
            # Verifica che almeno una cartella esista
            existing_folders = []
            for folder in folders_to_include:
                folder_path = os.path.join(model_dir, folder)
                if os.path.exists(folder_path):
                    existing_folders.append(folder)
                    print(f"  ✅ Found: {folder}")
                else:
                    print(f"  ⚠️ Missing: {folder}")
            
            if not existing_folders:
                print(f"❌ No valid folders found to include in ZIP")
                return False
            
            # Crea lo ZIP
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                total_files = 0
                
                for folder in existing_folders:
                    folder_path = os.path.join(model_dir, folder)
                    print(f"📂 Adding folder to ZIP: {folder}")
                    
                    # Aggiungi tutti i file dalla cartella
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            # Percorso completo del file
                            file_path = os.path.join(root, file)
                            
                            # Percorso relativo dalla directory del modello
                            relative_path = os.path.relpath(file_path, model_dir)
                            
                            # Aggiungi il file allo ZIP mantenendo la struttura
                            zipf.write(file_path, relative_path)
                            total_files += 1
                            
                            # Log solo alcuni file per non intasare i log
                            if total_files <= 10 or total_files % 100 == 0:
                                print(f"  📄 Added: {relative_path}")
                
                print(f"📊 Total files added to ZIP: {total_files}")
            
            # Verifica che il ZIP sia stato creato
            if os.path.exists(zip_path):
                zip_size = os.path.getsize(zip_path)
                
                # Calcola dimensione originale per rapporto compressione
                original_size = 0
                for folder in existing_folders:
                    folder_path = os.path.join(model_dir, folder)
                    original_size += self.get_folder_size(folder_path)
                
                compression_ratio = (zip_size / original_size * 100) if original_size > 0 else 0
                
                print(f"✅ Training ZIP created successfully")
                print(f"📦 ZIP size: {zip_size / 1024 / 1024:.2f} MB")
                print(f"📁 Original size: {original_size / 1024 / 1024:.2f} MB")
                print(f"🗜️ Compression ratio: {compression_ratio:.1f}%")
                return True
            else:
                print(f"❌ ZIP file was not created")
                return False
                
        except Exception as e:
            print(f"❌ Error creating training ZIP: {e}")
            return False

    def get_folder_size(self,folder_path):
        """
        Calcola la dimensione totale di una cartella.
        
        Args:
            folder_path: Percorso della cartella
            
        Returns:
            int: Dimensione in bytes
        """
        total_size = 0
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except Exception as e:
            print(f"⚠️ Warning: Error calculating folder size for {folder_path}: {e}")
        
        return total_size
    
    def download_and_extract_phase_zip(self, zip_s3_key, extract_to_dir):
        """
        Scarica un ZIP della fase da S3 e lo estrae nella directory specificata.
        
        Args:
            zip_s3_key: Chiave S3 del file ZIP
            extract_to_dir: Directory dove estrarre i contenuti
            
        Returns:
            bool: True se l'operazione è riuscita
        """
        try:
            print(f"📦 Downloading phase ZIP from S3: {zip_s3_key}")
            
            # Usa una directory temporanea per scaricare il ZIP
            with tempfile.TemporaryDirectory() as temp_dir:
                local_zip_path = os.path.join(temp_dir, "phase.zip")
                
                # Scarica il ZIP da S3
                repository_service.download(zip_s3_key, local_zip_path)
                
                if not os.path.exists(local_zip_path):
                    print(f"❌ Failed to download ZIP from S3")
                    return False
                
                zip_size = os.path.getsize(local_zip_path)
                print(f"✅ ZIP downloaded successfully: {zip_size / 1024 / 1024:.2f} MB")
                
                # Estrai il ZIP
                print(f"📂 Extracting ZIP to: {extract_to_dir}")
                
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    # Lista i contenuti del ZIP per debug
                    zip_contents = zip_ref.namelist()
                    print(f"📋 ZIP contains {len(zip_contents)} files")
                    
                    # Mostra alcuni file di esempio
                    example_files = zip_contents[:3]
                    if len(zip_contents) > 3:
                        example_files.append("...")
                    print(f"📄 Example files: {example_files}")
                    
                    # Estrai tutti i file
                    zip_ref.extractall(extract_to_dir)
                    
                print(f"✅ ZIP extracted successfully")
                
                # Verifica che l'estrazione sia andata a buon fine
                extracted_files = []
                for root, dirs, files in os.walk(extract_to_dir):
                    extracted_files.extend(files)
                
                print(f"📊 Extracted {len(extracted_files)} files total")
                return True
                
        except Exception as e:
            print(f"❌ Error downloading/extracting ZIP: {e}")
            return False
    
    def create_phase_zip_and_upload(self,model_id:str,model_dir: str,zip_file_name: str,target_folders: []):

        with tempfile.TemporaryDirectory() as temp_dir:
                # Costruisci il nome del ZIP usando l'enum Phase
                zip_path = os.path.join(temp_dir, zip_file_name)
                success = self.create_phase_zip(model_dir, target_folders, zip_path)

                # 6. Upload risultati COLMAP su S3
                # Lista delle cartelle/file da uploadare
                
                if success and os.path.exists(zip_path):
                    zip_size = os.path.getsize(zip_path)
                    print(f"✅ Training ZIP created successfully: {zip_size / 1024 / 1024:.2f} MB")
                    
                    # Upload del ZIP nella cartella staging
                    zip_s3_key = f"{S3_STAGING_PREFIX}/{model_id}/{zip_file_name}"
                    repository_service.upload(zip_path, zip_s3_key)
                    print(f"✅ Phase ZIP caricato in staging: {zip_s3_key}")
                    
                    # Il file ZIP viene automaticamente cancellato quando esce dal with
                    print(f"🧹 Temporary  ZIP will be cleaned up automatically")
                else:
                    self.fail(model_id,"training",f"Error: Failed to create training phase ZIP for model_id {model_id}")


    def send_to_next_phase(self, model_id, next_queue, additional_data=None):
        """Invia il job alla fase successiva"""
        self.channel.basic_publish(
            exchange='',
            routing_key=next_queue,
            body=json.dumps({"model_id": model_id, "additional_data": additional_data})
        )


    def handle_exit(self, signum, frame):
        """Gestisce la chiusura dell'applicazione"""
        print("\n🛑 Closing application...")
        close_connection(self.connection)
        sys.exit(0)
