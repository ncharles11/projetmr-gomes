import os
import logging
import spotipy
from spotipy.oauth2 import SpotifyOAuth

logger = logging.getLogger(__name__)

class SpotifyManager:
    """Gère la connexion Spotify et les commandes de lecture."""

    def __init__(self):
        self.sp = None
        self._setup()

    def _setup(self):
        client_id     = os.environ.get("SPOTIPY_CLIENT_ID")
        client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
        redirect_uri  = os.environ.get("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback")

        if not client_id or not client_secret:
            logger.warning("Variables SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET absentes. Spotify désactivé.")
            return
        try:
            auth = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-modify-playback-state user-read-playback-state",
                open_browser=False,
            )
            self.sp = spotipy.Spotify(auth_manager=auth)
            logger.info("SpotifyManager initialisé.")
        except Exception as e:
            logger.error(f"Échec initialisation Spotify : {e}")

    def _ready(self) -> bool:
        return self.sp is not None

    def play_playlist(self, playlist_id: str):
        if not self._ready() or not playlist_id:
            return
        try:
            self.sp.start_playback(context_uri=f"spotify:playlist:{playlist_id}")
            logger.info(f"Playlist lancée : {playlist_id}")
        except Exception as e:
            logger.warning(f"play_playlist : {e}")

    def toggle_play_pause(self):
        if not self._ready():
            return
        try:
            pb = self.sp.current_playback()
            if pb and pb.get("is_playing"):
                self.sp.pause_playback()
            else:
                self.sp.start_playback()
        except Exception as e:
            logger.warning(f"toggle_play_pause : {e}")

    def next_track(self):
        if not self._ready():
            return
        try:
            self.sp.next_track()
        except Exception as e:
            logger.warning(f"next_track : {e}")

    def prev_track(self):
        if not self._ready():
            return
        try:
            self.sp.previous_track()
        except Exception as e:
            logger.warning(f"prev_track : {e}")

    def get_current_track_info(self) -> dict:
        default = {"title": "En attente...", "artist": "—", "is_playing": False}
        if not self._ready():
            return default
        try:
            pb = self.sp.current_playback()
            if not pb or not pb.get("item"):
                return default
            item = pb["item"]
            return {
                "title":      item.get("name", "Inconnu"),
                "artist":     ", ".join(a["name"] for a in item.get("artists", [])),
                "is_playing": pb.get("is_playing", False),
            }
        except Exception as e:
            logger.warning(f"get_current_track_info : {e}")
            return default
