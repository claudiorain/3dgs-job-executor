import subprocess
import threading
import time
import json
import os
from pathlib import Path
from typing import Dict, Optional


class VRAMMonitor:
    def __init__(self, interval: float = 3.0, log_directory: Optional[str] = None):
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.vram_samples = []
        self.max_vram = 0
        self.start_time = None
        self.log_directory = log_directory
        self.actual_log_path = None
        self.log_file = None
            
    def _setup_log_file(self):
        """Crea il file di log quando inizia il monitoring"""
        if not self.log_directory:
            return
            
        try:
            # Crea la directory se non esiste
            log_dir = Path(self.log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Genera nome file con timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"vram_monitor_{timestamp}.csv"
            self.actual_log_path = log_dir / filename
            
            # Apre il file per scrittura
            self.log_file = open(self.actual_log_path, 'w')
            
            # Scrive header CSV
            self.log_file.write("timestamp,elapsed_seconds,used_mb,total_mb,gpu_util,temp_c\n")
            self.log_file.flush()
            
            print(f"📝 VRAM log will be saved to: {self.actual_log_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not setup log file in {self.log_directory}: {e}")
            self.log_file = None
            self.actual_log_path = None

    def get_gpu_memory(self) -> Optional[Dict]:
        """Get current GPU memory usage via nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                data = lines[0].split(', ')
                return {
                    'used_mb': int(data[0]),
                    'total_mb': int(data[1]),
                    'gpu_util': int(data[2]),
                    'temp_c': int(data[3]),
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"⚠️ Error getting GPU memory: {e}")
            return None

    def _log_to_file(self, gpu_info: Dict):
        """Scrive i dati nel file di log"""
        if self.log_file and gpu_info:
            try:
                elapsed = gpu_info['timestamp'] - self.start_time
                line = f"{gpu_info['timestamp']:.1f},{elapsed:.1f},{gpu_info['used_mb']},{gpu_info['total_mb']},{gpu_info['gpu_util']},{gpu_info['temp_c']}\n"
                self.log_file.write(line)
                self.log_file.flush()  # Force write to disk
            except Exception as e:
                print(f"⚠️ Error writing to log file: {e}")

    def _monitor_loop(self):
        """Loop di monitoring in background"""
        while self.monitoring:
            gpu_info = self.get_gpu_memory()
            if gpu_info:
                self.vram_samples.append(gpu_info)
                self.max_vram = max(self.max_vram, gpu_info['used_mb'])
                
                # Log su file se abilitato
                self._log_to_file(gpu_info)
                
                # Log ogni 20 samples (~1 minuto con interval=3s)
                if len(self.vram_samples) % 20 == 0:
                    avg_vram = sum(s['used_mb'] for s in self.vram_samples[-20:]) / 20
                    elapsed_min = (time.time() - self.start_time) / 60
                    log_msg = f"🔧 [{elapsed_min:.1f}min] VRAM: {gpu_info['used_mb']}MB | Avg: {avg_vram:.0f}MB | Peak: {self.max_vram}MB | GPU: {gpu_info['gpu_util']}% | Temp: {gpu_info['temp_c']}°C"
                    print(log_msg)
                
            time.sleep(self.interval)

    def start_monitoring(self):
        """Avvia il monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.start_time = time.time()
            self.vram_samples = []
            self.max_vram = 0
            
            # Setup del file di log DOPO aver iniziato il monitoring
            self._setup_log_file()
            
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            
            log_info = f" (logging to {self.actual_log_path})" if self.actual_log_path else ""
            print(f"🚀 VRAM monitoring started{log_info}")

    def stop_monitoring(self) -> Dict:
        """Ferma il monitoring e ritorna le statistiche"""
        if self.monitoring:
            self.monitoring = False
            if self.thread:
                self.thread.join(timeout=5)
            
            # Chiude il file di log
            if self.log_file:
                try:
                    self.log_file.close()
                    print(f"📝 VRAM log saved to: {self.actual_log_path}")
                except Exception as e:
                    print(f"⚠️ Error closing log file: {e}")
                finally:
                    self.log_file = None
            
            if not self.vram_samples:
                return {"error": "No VRAM data collected"}
            
            # Calcola statistiche finali
            vram_values = [s['used_mb'] for s in self.vram_samples]
            gpu_util_values = [s['gpu_util'] for s in self.vram_samples]
            temp_values = [s['temp_c'] for s in self.vram_samples]
            
            stats = {
                'peak_vram_mb': self.max_vram,
                'avg_vram_mb': round(sum(vram_values) / len(vram_values), 1),
                'min_vram_mb': min(vram_values),
                'avg_gpu_utilization': round(sum(gpu_util_values) / len(gpu_util_values), 1),
                'max_temperature_c': max(temp_values),
                'monitoring_duration_s': round(time.time() - self.start_time, 1),
                'total_samples': len(self.vram_samples),
                'log_file_path': str(self.actual_log_path) if self.actual_log_path else None
            }
            
            print(f"📊 Final VRAM Stats - Peak: {stats['peak_vram_mb']}MB | Avg: {stats['avg_vram_mb']}MB | GPU Avg: {stats['avg_gpu_utilization']}%")
            
            return stats
            
        return {"error": "Monitoring was not active"}


# Esempio di utilizzo:
if __name__ == "__main__":
    # Monitor senza logging
    monitor1 = VRAMMonitor(interval=2.0)
    
    # Monitor con logging - passi la DIRECTORY, non il file
    monitor2 = VRAMMonitor(interval=3.0, log_directory="/tmp/vram_logs")
    
    # Avvia monitoring
    monitor2.start_monitoring()
    
    # Simula training per 10 secondi
    time.sleep(10)
    
    # Ferma e ottieni stats
    stats = monitor2.stop_monitoring()
    print(f"Risultati: {json.dumps(stats, indent=2)}")