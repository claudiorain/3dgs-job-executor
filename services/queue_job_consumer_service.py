from config.message_queue import get_connection,get_channel,close_connection  # Assicurati che questa funzione restituisca il client del database
from services.model_service import ModelService
from services.repository_service import RepositoryService
from utils.frame_extractor import FrameExtractor
from utils.gaussian_estimator import GaussianEstimator
from converters.ply_to_gsplat_converter import save_splat_file,process_ply_to_splat
from utils.gaussian_frame_segmentator import segment_folder

import os
import sys
import json
import asyncio
import requests
import shutil
import threading
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import tempfile

model_service = ModelService()
repository_service = RepositoryService()
frame_extractor = FrameExtractor()
gaussian_estimator = GaussianEstimator()

WORKING_DIR = os.getenv("MODEL_WORKING_DIR") 
GAUSSIAN_SPLATTING_API_URL = "http://gaussian-splatting-api:8050"
COLMAP_API_URL = "http://colmap-converter-api:8060"

# Creiamo la mappa che associa ciascun valore a un oggetto con api-url
engine_map = {
    'INRIA': { 'api-url': 'http://gaussian-splatting-api:8100' },
    'MCMC': { 'api-url': 'http://3dgs-mcmc-api:8101' },
    'TAMING': { 'api-url': 'http://taming-3dgs-api:8102' },
    'GSPLAT-INRIA': { 'api-url': 'http://nerfstudio-gsplat-api:8103' },
    'GSPLAT-MCMC': { 'api-url': 'http://nerfstudio-gsplat-api:8103' }
}

# Assicurati che la cartella esista
os.makedirs(WORKING_DIR, exist_ok=True)

class QueueJobService:

    def __init__(self):
        """Inizializza la connessione a RabbitMQ"""
        self.connection = get_connection()
        self.channel = get_channel(self.connection)

    async def download_and_process_video(self, ch, method, model_id, data):
        """
        Fase 1: Download del video e creazione dei fotogrammi con preprocessing SAM2
        """
        model = model_service.get_model_by_id(model_id)
        if not model:
            print(f"Errore: Nessun documento trovato per model_id {model_id}")
            return
                    
        video_s3_key = model.video_s3_key
        if not video_s3_key:
            print(f"Errore: Nessun video_s3_key trovato per model_id {model_id}")
            return
                    
        model_service.update_model_status(model_id, {"status": "VIDEO_PROCESSING"})

        model_dir = os.path.join(WORKING_DIR, f"{model_id}")
        os.makedirs(model_dir, exist_ok=True)

        # Crea una directory temporanea per scaricare il video
        with tempfile.TemporaryDirectory() as temp_video_dir:
            # Percorso del file video nella cache
            local_video_path = os.path.join(temp_video_dir, 'video.mp4')

            # Verifica se il video è già nella cache
            repository_service.download_or_cache_video(video_s3_key, local_video_path)

            # Calcola l'hash del video
            video_hash = frame_extractor.generate_video_hash(local_video_path)

            # Cartelle di lavoro per il video
            processed_video_dir = os.path.join(os.path.join(WORKING_DIR, 'processed_videos'), video_hash)
            frames_output_folder = os.path.join(processed_video_dir, 'input_orig')
            
            # 🆕 Cartella per frames preprocessati
            frames_preprocessed_folder = os.path.join(processed_video_dir, 'input_segmented')

            thumbnail_s3_key = f"models/{model_id}/thumbnail.jpg"
            
            if os.path.exists(processed_video_dir):
                print(f"Video già processato. Utilizzo i dati esistenti per l'hash: {video_hash}")
                
                # Controlla se abbiamo anche i frames preprocessati
                if os.path.exists(frames_preprocessed_folder):
                    print(f"✅ Frames preprocessati trovati nella cache")
                    use_preprocessed = True
                else:
                    print(f"🔄 Cache trovata ma mancano frames preprocessati, rigenerazione...")
                    use_preprocessed = False
                    
            else:
                print(f"Video non trovato nella cache, elaborazione in corso...")
                # Crea le cartelle necessarie
                os.makedirs(processed_video_dir, exist_ok=True)
                os.makedirs(frames_output_folder, exist_ok=True)
                os.makedirs(frames_preprocessed_folder, exist_ok=True)
                
                # Procedi con il processamento del video e creazione dei fotogrammi
                thumbnail_path = self.process_video(local_video_path, frames_output_folder)
                
                print(f"✅ Thumbnail local path: {thumbnail_path} and exists? " + str(os.path.exists(thumbnail_path)))
                if thumbnail_path and os.path.exists(thumbnail_path):
                    try:
                        repository_service.upload(thumbnail_path, thumbnail_s3_key)
                        print(f"✅ Thumbnail caricata su S3: {thumbnail_s3_key}")
                    except Exception as e:
                        print(f"❌ Errore durante l'upload della thumbnail su S3: {e}")
                        thumbnail_s3_key = None
                
                use_preprocessed = False
            
            # 🆕 PREPROCESSING DEI FRAMES CON SAM2
            """if not use_preprocessed:
                print(f"🤖 Avvio preprocessing frames con SAM2...")
                
                try:
                    # Aggiorna status per indicare preprocessing in corso
                    model_service.update_model_status(model_id, {
                        "status": "FRAME_PREPROCESSING", 
                        "thumbnail_s3_key": thumbnail_s3_key
                    })
                    
                    # Segmentazione piante
                    results = segment_folder(
                        input_folder=frames_output_folder,
                        output_folder=frames_preprocessed_folder,
                        text_prompt=model.description,
                        box_threshold=0.5,
                        text_threshold=0.35,
                        save_debug=False
                    )
 
                    print(f"Successo: {results['successful']}/{results['total']}")

                    # Se il preprocessing ha fallito su troppe immagini, usa frames originali
                    success_rate = results['successful'] / results['total']
                    if success_rate < 0.5:  # Meno del 50% successo
                        print(f"⚠️  Tasso successo preprocessing troppo basso ({success_rate:.1%}), uso frames originali")
                        frames_to_use = frames_output_folder
                        preprocessing_used = False
                    else:
                        frames_to_use = frames_preprocessed_folder
                        preprocessing_used = True
                    
                except Exception as e:
                    print(f"❌ Errore durante preprocessing: {e}")
                    print(f"🔄 Fallback su frames originali")
                    frames_to_use = frames_output_folder
                    preprocessing_used = False
                    
                    # Log dell'errore nel modello
                    model_service.update_model_status(model_id, {
                        "preprocessing_error": str(e),
                        "frames_preprocessed": False
                    })
            else:
                # Usa frames preprocessati dalla cache
                frames_to_use = frames_preprocessed_folder
                preprocessing_used = True
                print(f"✅ Utilizzo frames preprocessati dalla cache")

            # 🆕 Aggiorna il path dei frames per il prossimo step
            print(f"📁 Frames finali utilizzati: {frames_to_use}")
            print(f"🎯 Preprocessing utilizzato: {preprocessing_used}")"""

        frames_to_use = frames_output_folder
        # Passa al prossimo job con informazioni aggiuntive
        model_service.update_model_status(model_id, {
            "status": "POINT_CLOUD_RECONSTRUCTION", 
            "thumbnail_s3_key": thumbnail_s3_key,
            "frames_path": frames_to_use,
            "preprocessing_used": False
        })
        
        self.send_to_next_phase(model_id, "point_cloud_queue", video_hash)

    def get_preprocessing_config(self, model):
        """
        Ottieni configurazione preprocessing dal modello o usa defaults.
        """
        # Puoi personalizzare in base al modello o alle sue properties
        default_config = {
            "method": "center_point",  # o "auto_bbox" per rilevamento automatico
            "removal_level": "moderate",  # conservative, moderate, aggressive
            "transparent_bg": True,  # Raccomandato per Gaussian Splatting
            "model_size": "base_plus"  # tiny, small, base_plus, large
        }
        
        # Se il modello ha configurazioni specifiche, le usi
        if hasattr(model, 'preprocessing_config') and model.preprocessing_config:
            config = default_config.copy()
            config.update(model.preprocessing_config)
            return config
        
        return default_config

    def process_video(self, local_video_path, frames_output_folder):
        # Estrarre fotogrammi dal video
        return  frame_extractor.extract_frames(local_video_path, frames_output_folder,
                                                        threshold=0.40,
                                                        min_contour_area=500,
                                                        max_frames=200,  # Extract more frames for 360° scenes
                                                        min_sharpness=100,
                                                        enforce_temporal_distribution=True,
                                                        downscale_factor=1)  # Pro

    async def build_point_cloud(self, ch, method, model_id, data):
        """Fase 2: Creazione del punto nuvola tramite Gaussian Splatting"""

        video_hash = data.get("additional_data")
        processed_video_dir = os.path.join(os.path.join(WORKING_DIR, 'processed_videos'), video_hash)
        
        # 🆕 Salva backup delle immagini segmentate PRIMA di COLMAP
        input_dir = os.path.join(processed_video_dir, 'input')

        print(f"✅ Input dir {input_dir} esiste!")
        # Prepara directory input per COLMAP
        self.prepare_input_directory(processed_video_dir)
        
        # Cartella sparse per la point cloud
        sparse_dir = os.path.join(processed_video_dir, 'sparse')

        if os.path.exists(sparse_dir):
            print(f"Nuvola di punti già esistente per model_id {model_id}. Saltando la generazione.")
        else:
            print(f"Generando la nuvola di punti per model_id {model_id}...")
            
            # COLMAP API call
            convert_request = {"input_dir": processed_video_dir}
            response = requests.post(COLMAP_API_URL + "/convert", json=convert_request)
            
            if response.status_code != 200:
                print(f"❌ Errore COLMAP: {response.text}")
                return
            
            # 🆕 SOSTITUZIONE IMMEDIATA DOPO COLMAP
            images_dir = os.path.join(processed_video_dir, 'images')
            print(f"✅ Input dir {images_dir} esiste!")
           
            if not os.path.exists(sparse_dir):
                print(f"Errore: la nuvola di punti non è stata creata correttamente per model_id {model_id}")
                return
            
            print(f"Nuvola di punti generata con successo per model_id {model_id}.")

        # Passa al prossimo job (model_training)
        model_service.update_model_status(model_id, {"status": "MODEL_TRAINING"})

        model = model_service.get_model_by_id(model_id)
        if not model:
            print(f"Errore: Nessun documento trovato per model_id {model_id}")
            return
        
        self.send_to_next_phase(model_id, "model_training_queue", {"engine": model.engine, "input_dir": processed_video_dir})

    
    def prepare_input_directory(self, processed_video_dir):
        """
        Prepara SOLO la directory 'input/' per COLMAP.
        Priorità: segmented > orig
        Rimuove TUTTO il resto per evitare confusione.
        """
        import shutil
        
        segmented_dir = os.path.join(processed_video_dir, 'input_segmented')
        orig_dir = os.path.join(processed_video_dir, 'input_orig')
        input_dir = os.path.join(processed_video_dir, 'input')
        
        print(f"🔧 Preparazione input/ per COLMAP...")
        
        # Scegli la fonte migliore
        if os.path.exists(segmented_dir):
            source_dir = segmented_dir
            frames_type = "segmentati"
        elif os.path.exists(orig_dir):
            source_dir = orig_dir
            frames_type = "originali"
        else:
            print(f"❌ Nessuna immagine trovata!")
            return False
        
        print(f"📁 Usando frames {frames_type}")
        
        # Rinomina la directory scelta in 'input'
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        shutil.move(source_dir, input_dir)
        
        # Rimuovi TUTTO il resto
        for item in os.listdir(processed_video_dir):
            if item == 'input':
                continue
            item_path = os.path.join(processed_video_dir, item)
            if os.path.isdir(item_path):
                print(f"🗑️  Rimozione {item}/")
                shutil.rmtree(item_path)
        
        # Conta risultato
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"✅ COLMAP userà {len(files)} immagini {frames_type}")

    async def train_model(self, ch, method, model_id, data):
        print(f"start training model")
        """Fase 3: Addestramento del modello"""
        input_dir = data.get("additional_data").get("input_dir")
        estimated_cap_max = gaussian_estimator.estimate_from_colmap(
                input_dir, 
                density_factor=6.0,
                min_gaussians=300000,
                max_gaussians=5000000
            )
    
        print(f"estimated max gaussians " + str(estimated_cap_max))

        # Chiamata API per il training del modello
        train_output_folder = os.path.join(os.path.join(WORKING_DIR, f"{model_id}"), 'output')
    
        engine = data.get("additional_data").get("engine")

        train_request = {"input_dir": input_dir, "output_dir": train_output_folder, "cap_max": int(estimated_cap_max * 1.2),
                        "train_type": {"GSPLAT-INRIA": "default", "GSPLAT-MCMC": "mcmc"}.get(engine, "default") }#, "cap_max":estimated_cap_max
        response = requests.post(engine_map.get(engine).get('api-url') + "/train", json=train_request)
        if response.status_code != 200:
            print(f"Errore durante il training: {response.text}")
            return
        print(f"training ended")

        # Passa al prossimo job (upload)
        self.send_to_next_phase(model_id, "upload_queue", train_output_folder)

    async def upload_model(self, ch, method, model_id, data):
        """Fase 4: Caricamento del modello addestrato su S3"""
        try:
            model = model_service.get_model_by_id(model_id)
            with tempfile.TemporaryDirectory() as temp_dir:
                # Crea i percorsi come al solito, ma per gsplat_path utilizziamo la cartella temporanea
                model_dir = os.path.join(WORKING_DIR, f"{model_id}")
                train_output_folder = data.get("additional_data")

                if model.engine in ["INRIA","MCMC","TAMING"]:  # Gruppo 1: INRIA, TAMING scrivono in modo simile
                    ply_path = os.path.join(train_output_folder, "point_cloud/iteration_30000/point_cloud.ply")
                    cameras_file_path = os.path.join(train_output_folder, "cameras.json")

                    # Usa la cartella temporanea per il file .splat
                    gsplat_path = os.path.join(temp_dir, "point_cloud.splat")
                    # Processa il file .ply e salva il .splat nella cartella temporanea
                    save_splat_file(process_ply_to_splat(ply_path), gsplat_path)

                    # Aggiungi il file cameras.json alla cartella temporanea
                    shutil.copy(cameras_file_path, temp_dir)
                elif model.engine in ["GSPLAT-INRIA", "GSPLAT-MCMC"]:
                    ply_path = os.path.join(train_output_folder, "ply/point_cloud_29999.ply")
                    # Usa la cartella temporanea per il file .splat
                    gsplat_path = os.path.join(temp_dir, "point_cloud.splat")
                    # Processa il file .ply e salva il .splat nella cartella temporanea
                    save_splat_file(process_ply_to_splat(ply_path), gsplat_path)

                # Creazione del file ZIP che contiene il file .splat e il file cameras.json
                zip_filename = os.path.join(model_dir, "3d_model.zip")
                shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', temp_dir)

                # Inizio del caricamento del modello su S3
                print(f"Inizio del caricamento del modello {model_id} su S3...")
                s3_key = f"models/{model_id}/3d_model.zip"
                try:
                    repository_service.upload(zip_filename, s3_key)
                    print(f"✅ Il modello {model_id} è stato caricato con successo su S3 nella chiave {s3_key}")
                except Exception as e:
                    print(f"Errore durante l'upload su S3: {e}")
                    model_service.update_model_status(model_id, {"status": "FAILED"})
                    return
                
                model_service.update_model_status(model_id, {"status": "METRICS_GENERATION", "output_s3_key": s3_key})
                print(f"Model {model_id} caricato su S3 con successo!")
                # Passa al prossimo job (upload)
                self.send_to_next_phase(model_id, "metrics_generation_queue", train_output_folder)
                
        except FileNotFoundError as e:
            print(f"❌ Errore: {e}")
            model_service.update_model_status(model_id, {"status": "ERROR", "error_message": str(e)})
        except NoCredentialsError:
            print("❌ Errore: Credenziali AWS mancanti.")
            model_service.update_model_status(model_id, {"status": "ERROR", "error_message": "AWS credentials missing"})
        except PartialCredentialsError:
            print("❌ Errore: Credenziali AWS parziali.")
            model_service.update_model_status(model_id, {"status": "ERROR", "error_message": "Partial AWS credentials"})
        except Exception as e:
            print(f"❌ Errore durante il caricamento del modello: {e}")
            model_service.update_model_status(model_id, {"status": "ERROR", "error_message": str(e)})
        
    async def generate_metrics(self, ch, method, model_id, data):
        """Fase 5: Generazione delle metriche e salvataggio su Mongo"""
        # Aggiorna lo stato del modello a COMPLETED
        try:
            model = model_service.get_model_by_id(model_id)

            train_output_folder = data.get("additional_data")
            ssim = None
            psnr = None
            lpips = None
            if model.engine in ["INRIA","MCMC","TAMING"]:  # Gruppo 1: INRIA, TAMING scrivono in modo simile
                render_request = { "output_dir": train_output_folder}
                response = requests.post(engine_map.get(model.engine).get('api-url') + "/render", json=render_request)
                response.raise_for_status()  # Controlla che non ci siano errori nel render

                metrics_request = { "output_dir": train_output_folder}
                response = requests.post(engine_map.get(model.engine).get('api-url') + "/metrics", json=metrics_request)
                response.raise_for_status()  # Controlla che non ci siano errori nel render

                # Verifica che il file results.json esista
                results_json_path = os.path.join(train_output_folder, "results.json")
                if not os.path.exists(results_json_path):
                    raise FileNotFoundError("Il file 'results.json' non è stato trovato.")
                # Leggi il contenuto di results.json
                with open(results_json_path, 'r') as f:
                    results_data = json.load(f)
        
                # Estrai le metriche necessarie
                if "ours_30000" in results_data:
                    metrics = results_data["ours_30000"]
                    ssim = metrics.get("SSIM", None)
                    psnr = metrics.get("PSNR", None)
                    lpips = metrics.get("LPIPS", None)
                else:
                    raise KeyError("La sezione 'ours_30000' non è presente nel file results.json.")
            elif model.engine in ["GSPLAT-INRIA", "GSPLAT-MCMC"]:
               # Verifica che il file results.json esista
                results_json_path = os.path.join(train_output_folder, "stats/val_step29999.json")

                # Leggi il contenuto di results.json
                with open(results_json_path, 'r') as f:
                    results_data = json.load(f)
                ssim = results_data.get("ssim", None)
                psnr = results_data.get("psnr", None)
                lpips = results_data.get("lpips", None)
                # Preparazione del dato per il database
                # Estrai le metriche necessarie
                
            results = {
                    "ssim": ssim,
                    "psnr": psnr,
                    "lpips": lpips
            }
            model_service.update_model_status(model_id, {"status": "COMPLETED", "results": results})
            print(f"Model {model_id} caricato su S3 con successo!")
    
        except FileNotFoundError as e:
            print(f"❌ Errore: {e}")
            model_service.update_model_status(model_id, {"status": "ERROR", "error_message": str(e)})
        except NoCredentialsError:
            print("❌ Errore: Credenziali AWS mancanti.")
            model_service.update_model_status(model_id, {"status": "ERROR", "error_message": "AWS credentials missing"})
        except PartialCredentialsError:
            print("❌ Errore: Credenziali AWS parziali.")
            model_service.update_model_status(model_id, {"status": "ERROR", "error_message": "Partial AWS credentials"})
        except Exception as e:
            print(f"❌ Errore durante il caricamento del modello: {e}")
            model_service.update_model_status(model_id, {"status": "ERROR", "error_message": str(e)})



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
