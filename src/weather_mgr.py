import requests
import logging

logger = logging.getLogger(__name__)

# Traduction des codes météo Open-Meteo → texte + icône
_WEATHER_CODES: dict[int, str] = {
    0:  "☀️ Grand soleil",
    1:  "🌤 Peu nuageux",
    2:  "⛅ Partiellement nuageux",
    3:  "☁️ Couvert",
    45: "🌫 Brouillard",
    48: "🌫 Brouillard givrant",
    51: "🌦 Bruine légère",
    53: "🌦 Bruine",
    55: "🌧 Bruine forte",
    61: "🌧 Pluie légère",
    63: "🌧 Pluie",
    65: "🌧 Pluie forte",
    71: "🌨 Neige légère",
    73: "🌨 Neige",
    75: "❄️ Neige forte",
    80: "🌦 Averses légères",
    81: "🌧 Averses",
    82: "⛈ Averses fortes",
    95: "⛈ Orage",
    96: "⛈ Orage + grêle",
    99: "⛈ Orage violent",
}


def check_internet_connection() -> bool:
    """Vérifie si une connexion Internet est disponible (timeout 1s)."""
    try:
        requests.head("http://www.google.com", timeout=1.0)
        return True
    except Exception:
        return False


def get_current_location() -> dict | None:
    """Position fixe temporaire — Badevel (évite la localisation IP incorrecte via 4G)."""
    return {"city": "Badevel", "lat": 47.5002, "lon": 6.9404}


def get_weather(lat: float, lon: float) -> str:
    """Retourne une chaîne lisible ex: '19°C ⛅ Partiellement nuageux'."""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        r = requests.get(url, timeout=5.0)
        r.raise_for_status()
        cw   = r.json().get("current_weather", {})
        temp = cw.get("temperature", "--")
        code = int(cw.get("weathercode", -1))
        desc = _WEATHER_CODES.get(code, "")
        return f"{temp}°C  {desc}"
    except Exception as e:
        logger.warning(f"get_weather : {e}")
        return "--°C"
