import logging
import socket
import subprocess
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_BT_NAME = socket.gethostname()   # nom Bluetooth = hostname du Pi (ex: "mehari")


class BluetoothManager:
    """Gère l'appairage A2DP (BlueZ/bluetoothctl) et le contrôle AVRCP (playerctl).

    Architecture :
      - Le Pi agit comme Sink A2DP : il reçoit l'audio du téléphone.
      - playerctl lit les métadonnées AVRCP exposées par BlueZ.
      - Toutes les commandes système sont non-bloquantes (threads démons).
    """

    def __init__(self):
        self._playerctl_ok    = self._check_cmd("playerctl")
        self._bluetoothctl_ok = self._check_cmd("bluetoothctl")
        if not self._playerctl_ok:
            logger.warning("playerctl introuvable — sudo apt install playerctl")
        if not self._bluetoothctl_ok:
            logger.warning("bluetoothctl introuvable — vérifiez l'installation de BlueZ")

    # ── Utilitaires ───────────────────────────────────────────────────────────

    @staticmethod
    def _check_cmd(cmd: str) -> bool:
        try:
            subprocess.run(["which", cmd], capture_output=True, check=True, timeout=2)
            return True
        except Exception:
            return False

    def _run(self, *cmd, timeout: int = 3) -> str:
        """Commande synchrone — retourne stdout ou '' en cas d'erreur."""
        try:
            r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except FileNotFoundError:
            return ""
        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout : {' '.join(cmd)}")
            return ""
        except Exception as e:
            logger.debug(f"_run {cmd} : {e}")
            return ""

    def _run_async(self, *cmd):
        """Fire-and-forget dans un thread démon."""
        threading.Thread(target=self._run, args=cmd, daemon=True).start()

    # ── Appairage A2DP ────────────────────────────────────────────────────────

    def make_discoverable(self, on_done: Optional[Callable[[str], None]] = None):
        """Rend le Pi découvrable + pair-able en arrière-plan.
        Appelle on_done(status_str) depuis le thread worker à la fin.
        """
        if not self._bluetoothctl_ok:
            if on_done:
                on_done("bluetoothctl introuvable — vérifiez BlueZ.")
            return

        def _worker():
            cmds = (
                "power on\n"
                "agent NoInputNoOutput\n"
                "default-agent\n"
                "discoverable on\n"
                "pairable on\n"
            )
            try:
                proc = subprocess.Popen(
                    ["bluetoothctl"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                proc.communicate(input=cmds.encode(), timeout=6)
                status = f"Visible comme « {_BT_NAME} » — connectez votre téléphone"
                logger.info("Bluetooth : mode découvrable activé.")
            except Exception as e:
                status = f"Erreur Bluetooth : {e}"
                logger.warning(status)
            if on_done:
                on_done(status)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Contrôles AVRCP ───────────────────────────────────────────────────────

    def toggle_play_pause(self):
        if self._playerctl_ok:
            self._run_async("playerctl", "play-pause")

    def next_track(self):
        if self._playerctl_ok:
            self._run_async("playerctl", "next")

    def prev_track(self):
        if self._playerctl_ok:
            self._run_async("playerctl", "previous")

    # ── Métadonnées ───────────────────────────────────────────────────────────

    def get_current_track_info(self) -> dict:
        """Retourne titre/artiste/état + indicateur de connexion via playerctl.
        DOIT être appelé depuis un thread — subprocess bloquant.

        Champs retournés :
          connected  : bool — True si un lecteur BT est actif
          is_playing : bool — True si l'état est "Playing"
          title      : str
          artist     : str
          art_url    : str — URL mpris:artUrl (file:// ou https://) ou ''
        """
        default = {
            "connected":  False,
            "title":      "Pas de musique en cours",
            "artist":     "—",
            "is_playing": False,
            "art_url":    "",
        }
        if not self._playerctl_ok:
            return default

        # playerctl status : code 0 si un lecteur existe, non-0 sinon
        try:
            r = subprocess.run(
                ["playerctl", "status"],
                capture_output=True, text=True, timeout=2,
            )
            status    = r.stdout.strip()
            connected = (r.returncode == 0 and bool(status))
        except Exception:
            return default

        if not connected:
            return default

        title   = self._run("playerctl", "metadata", "title",        timeout=2)
        artist  = self._run("playerctl", "metadata", "artist",       timeout=2)
        art_url = self._run("playerctl", "metadata", "mpris:artUrl", timeout=2)
        return {
            "connected":  True,
            "title":      title   or "Pas de musique en cours",
            "artist":     artist  or "—",
            "is_playing": status.lower() == "playing",
            "art_url":    art_url or "",
        }
