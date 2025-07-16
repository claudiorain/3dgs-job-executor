import tempfile
import os
from services.repository_service import RepositoryService
from services.model_service import ModelService
import zipfile

S3_STAGING_PREFIX = os.getenv('S3_STAGING_PREFIX', 'staging')

repository_service = RepositoryService()
model_service = ModelService()

class PhaseZipHelper:

    
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
            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                print(f"✅ Training ZIP created successfully")
                return True
            else:
                print(f"❌ ZIP file was not created")
                return False
                
        except Exception as e:
            print(f"❌ Error creating training ZIP: {e}")
            return False
        
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
                    return True
                else:
                    return False

