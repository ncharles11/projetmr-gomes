import serial
import json
import logging
import time
from src import config

logger = logging.getLogger(__name__)

class ESP32Communicator:
    def __init__(self):
        self.ser = None
        try:
            self.ser = serial.Serial(config.SERIAL_PORT, config.SERIAL_BAUDRATE, timeout=0.01)
            # Forcer le réveil de l'ESP32 sous macOS (toggle DTR/RTS)
            self.ser.dtr = False
            self.ser.rts = False
            time.sleep(0.1)
            self.ser.dtr = True
            self.ser.rts = True
            logger.info(f"Connected to ESP32 on {config.SERIAL_PORT} at {config.SERIAL_BAUDRATE} baud (timeout=0.01).")
        except Exception as e:
            logger.error(f"Failed to connect to ESP32 on {config.SERIAL_PORT}: {e}")

    def send_auth_data(self, person_id, temps_droite=0, temps_bas=0):
        if self.ser is None or not self.ser.is_open:
            logger.warning("Serial connection is not open. Cannot send data.")
            return

        if person_id is not None:
            data = {
                "status": "authenticated", 
                "user": person_id, 
                "tempsDroite": temps_droite, 
                "tempsBas": temps_bas
            }
        else:
            data = {
                "status": "locked", 
                "user": None, 
                "tempsDroite": 0, 
                "tempsBas": 0
            }

        try:
            json_str = json.dumps(data) + '\n'
            self.ser.write(json_str.encode('utf-8'))
            logger.debug(f"Sent to ESP32: {json_str.strip()}")
            
            # Attendre une réponse immédiate si possible (optionnel, déjà géré par read_message si appelé ailleurs)
            # time.sleep(0.1)
            # if self.ser.in_waiting > 0:
            #     response = self.ser.readline().decode('utf-8', errors='ignore').strip()
            #     logger.info(f"ESP32 response: {response}")
        except Exception as e:
            logger.error(f"Error sending data to ESP32: {e}")

    def send_raw(self, data: str):
        """Envoie des caractères bruts sur le port série (ex: 'H', 'B', 'G', 'D')."""
        if self.ser is None or not self.ser.is_open:
            logger.warning("Serial connection not open. Cannot send raw data.")
            return
        try:
            self.ser.write(data.encode('utf-8'))
            logger.debug(f"Sent raw: {data!r}")
        except Exception as e:
            logger.error(f"Error sending raw data to ESP32: {e}")

    def send_step(self, direction_char: str, duration_ms: int):
        """Envoie un ordre de pas moteur : {"direction":"D","temps":100}."""
        if self.ser is None or not self.ser.is_open:
            logger.warning("Serial connection not open. Cannot send step command.")
            return
        try:
            payload = json.dumps({"direction": direction_char, "temps": duration_ms}) + '\n'
            self.ser.write(payload.encode('utf-8'))
            logger.debug(f"Sent step: direction={direction_char!r} temps={duration_ms}")
        except Exception as e:
            logger.error(f"Error sending step command to ESP32: {e}")

    def read_message(self):
        """Lit un message depuis le port série.
        Les lignes préfixées '[ESP32]' sont des logs de débogage firmware — ignorées par le parseur JSON.
        """
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    return None
                if line.startswith('[ESP32]'):
                    logger.info(f"ESP32 log: {line}")
                    return None
                logger.info(f"Received from ESP32: {line}")
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    logger.info(f"ESP32 text: {line}")
                    return None
        except Exception as e:
            logger.error(f"Error reading from ESP32: {e}")
        return None
