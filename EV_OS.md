# 🚗 EV-OS : Mehari Cockpit (French Version)

**EV-OS** est un tableau de bord tactile embarqué conçu pour une Méhari électrique. Cet écosystème fusionne reconnaissance faciale, contrôle moteur rétroviseur intelligent, accès média Bluetooth A2DP Sink et meteo en temps réel sur Raspberry Pi 5 avec écran tactile 10,1". Le projet incarne une philosophie d'autonomie technique : zero dépendance API cloud payante (Spotify Web API), contrôle local complet, et interface 100% Python non-bloquante via CustomTkinter.

**Caractère du projet** : ingénierie embarquée pragmatique. Pas de framework lourd, juste du Python efficace + BlueZ/PipeWire pour le Bluetooth, OpenCV/ONNX pour la vision, et une interface Tkinter moderne pensée pour le tactile.

---

## 🏗️ 1. Origine du Projet & Base de Départ

### Contexte Initial

Le projet a débuté par une démonstration de reconnaissance faciale en Python sur Raspberry Pi, avec une interface Tkinter basique. Trois composantes formaient l'épine dorsale :

1. **Détection faciale** : OpenCV + ONNX Runtime (`det_10g.onnx`), conversion d'images en embeddings haute-dimension via ResNet 50 (`w600k_r50.onnx`)
2. **Anti-spoofing** : PyTorch MiniFASNetV2 (`2.7_80x80_MiniFASNetV2.pth`) pour rejeter les attaques par photo/vidéo
3. **Reconnaissance** : comparaison cosinus des embeddings avec seuil configurable (0.50 par défaut)

### Architecture API Spotify Initiale (Limitation)

La v0.0.1 intégrait l'**API Spotify Web** pour la musique embarquée :
- ✅ Métadonnées : titre, artiste, couverture
- ⚠️ **Problèmes critiques** :
  - Authentification OAuth 2.0 + refresh token requise → infrastructure externe
  - Compte Premium obligatoire pour l'API (coût)
  - Pas d'accès audio local → dépendance au streaming cloud
  - Fragile en milieu embarqué avec connectivité intermittente

**Verdict** : inadapté à une Méhari électrique en autonomie.

### Pile d'Apprentissage Profond (Inchangée)

```python
# Modèles utilisés (config.py)
DETECTOR_MODEL = "det_10g.onnx"          # Détection visages
EMBEDDER_MODEL = "w600k_r50.onnx"        # Embeddings 512-D
ANTI_SPOOFING_MODEL = "2.7_80x80_MiniFASNetV2.pth"  # Liveness check
```

**Stack** : NumPy, OpenCV, ONNX Runtime (CPU/GPU), PyTorch, scikit-learn.

---

## 🛠️ 2. Évolutions & Améliorations Apportées

### ✨ Migration CustomTkinter & Design System EV-OS

**Avant** : Tkinter pur (gris/bleu par défaut, layout rigide).

**Après v0.1.0** : Refonte totale en **CustomTkinter 5.2+** avec thème dark propriétaire.

#### Palette Couleurs EV-OS (Cyberpunk + Pi5)

```python
# src/gui.py — Définition de la palette
BG       = '#0A1118'       # Fond noir ultra-profond
SIDEBAR  = '#0D141C'       # Sidebar légèrement plus clair
CARD     = '#16222F'       # Cartes / panneaux principaux
CARD_CTK = '#1B2B3B'       # Cartes CTkFrame (profondeur)
ENTRY_BG = '#0F1C28'       # Champs saisie / listbox
TEXT     = '#FFFFFF'       # Texte principal (blanc)
SUB      = '#94A3B8'       # Texte secondaire (gris bleuté)
CYAN     = '#00E5FF'       # Accent EV-OS (cyan électrique)
NAV_ACT  = '#1A3A4A'       # Navbouton actif
BLACK    = '#000000'       # Pour ombres/contrastes
```

Tous les composants (CTkFrame, CTkButton, CTkLabel, CTkProgressBar, CTkEntry) utilisent cette palette. 
**Résultat** : interface cohérente, moderne, lisible sur écran tactile 10,1" même en plein soleil.

#### Layout Non-Bloquant

- Tous les appels réseau, IA, et systèmes sont en **threads démons** indépendants
- GUI responsive même pendant rechero faciale ou connexion Bluetooth
- `threading.Thread(..., daemon=True)` partout
- **Boucle principale Tkinter** inchangée — juste poussée par `update()` via callbacks

### 🎮 Clavier Virtuel Tactile AZERTY – CTkVirtualKeyboard

**Problème résolu** : Wayland + XWayland sur Raspberry Pi 5 = conflits avec onboard/squeekboard (crashes, non-responsif).

**Solution** : Implémentation 100% Python CTkVirtualKeyboard embarquée (`src/gui.py`, lignes 71–186).

#### Architecture

```python
class CTkVirtualKeyboard(ctk.CTkFrame):
    """Clavier AZERTY CTkinter autonome, zéro dépendance système."""
    
    def show(self, target: ctk.CTkEntry, on_validate=None):
        """Affiche le clavier ancré en bas, cible l'entry."""
        self._target = target
        self._on_validate_cb = on_validate
        self.place(relx=0, rely=1.0, anchor='sw', relwidth=1.0)  # sticky bottom
        self.lift()
    
    def hide(self):
        self.place_forget()
    
    def _press(self, char: str):
        """Insère char dans la CTkEntry cible."""
        inner = self._target._entry  # tk.Entry sous-jacent
        inner.insert('insert', char)
    
    def _backspace(self):
        """Efface le caractère avant le curseur."""
        inner = self._target._entry
        pos = inner.index('insert')
        if pos > 0:
            inner.delete(pos - 1, pos)
```

#### Disposition AZERTY

```python
_AZERTY = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['A', 'Z', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['Q', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M'],
    ['W', 'X', 'C', 'V', 'B', 'N', '@', '.', '_', '-'],
]
```

Chaque touche est un `CTkButton` 48px de haut. Ligne inférieure : **Espace**, **⌫ (backspace rouge)**, **✓ OK (cyan)**. Pression sur **OK** appelle `on_validate_cb()` ou cache le clavier.

**Usage** :

```python
# Saisie du nom conducteur avec prompt overlay
def prompt_driver_name():
    dialog = ctk.CTkToplevel(self.root)
    entry = ctk.CTkEntry(dialog, ...)
    self._vkb.show(entry, on_validate=lambda: save_driver(entry.get()))
    self.root.wait_variable(...)  # Attend validation
```

### 🔧 Adaptation Raspberry Pi 5 & Production Kiosk

**Environnement cible** :
- Raspberry Pi 5 (Pi4 possible, plus lent)
- Labwc (composeur léger Wayland) + touchscreen 1280×800 ou 1024×600
- Port série USB → `/dev/ttyUSB0` pour ESP32 rétroviseur

#### Configuration Série ESP32 (hardware_comm.py)

```python
SERIAL_PORT = '/dev/ttyUSB0'  # Défini dans config.py
SERIAL_BAUDRATE = 115200      # Standard ESP32
```

Communication JSON bidirectionnelle :
```json
{
  "cmd": "move_mirror",
  "position": "up",
  "duration_ms": 500
}
```

#### Mode Kiosk & Fullscreen

```python
# src/gui.py
class FaceRecognitionGUI:
    def __init__(self, root: ctk.CTk):
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
```

Le kiosk démarre en plein écran. **Escape** bascule fullscreen (développement). En production, Escape verrouillé via udev/systemd.

#### Script de Démarrage (start_cockpit.sh)

```bash
#!/bin/bash
source .venv/bin/activate
DISPLAY=:0 python -m src.main
```

S'exécute au boot via crontab utilisateur ou systemd user service :
```bash
# ~/.config/systemd/user/cockpit.service (optionnel)
[Unit]
Description=EV-OS Cockpit Dashboard
After=multi-user.target

[Service]
Type=simple
ExecStart=/home/pi/projetmr-gomes/start_cockpit.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

### 🎵 Système Média Bluetooth A2DP Sink – Revolution Audio

**Problème v0.0.1** : API Spotify payante, pas d'autonomie audio locale.

**Nouvelle Architecture (v0.1.0+)** :

```
Téléphone (Spotify, YouTube Music, etc.)
    ↓ A2DP Stream
Raspberry Pi (BlueZ Sink)
    ↓ PipeWire (audio daemon)
    ↓ playerctl (AVRCP control + metadata)
GUI EV-OS
    ↓ Affiche cover art, titre, artiste
    ↓ Boutons play/pause/next/prev
```

#### Composante 1 : BluetoothManager (src/bluetooth_mgr.py)

Classe autonome gérant appairage + métadonnées.

**Appairage Découvrable** :

```python
def make_discoverable(self, on_done=None):
    """Rend Pi découvrable + pair-able en arrière-plan."""
    cmds = (
        "power on\n"
        "agent NoInputNoOutput\n"
        "default-agent\n"
        "discoverable on\n"
        "pairable on\n"
    )
    # Pipe `cmds` → bluetoothctl stdin (thread démon)
    # Appelle on_done(status_str) une fois terminé
```

Le Pi devient visible sous son hostname (ex: "mehari") pendant 3 min. Utilisateur appuie sur "Associer un téléphone", reste à la portée BT (3–10m selon Pi5 + antenne).

**Lecture Métadonnées AVRCP** :

```python
def get_current_track_info(self) -> dict:
    """Retourne titre/artiste/état + cover via playerctl."""
    # playerctl status → "Playing" | "Paused" | erreur
    # playerctl metadata title
    # playerctl metadata artist
    # playerctl metadata mpris:artUrl  ← file:// ou https://
    return {
        "connected":  bool,
        "title":      str,
        "artist":     str,
        "is_playing": bool,
        "art_url":    str,
    }
```

**Commandes AVRCP** (Fire-and-Forget) :

```python
def toggle_play_pause(self):
    self._run_async("playerctl", "play-pause")

def next_track(self):
    self._run_async("playerctl", "next")

def prev_track(self):
    self._run_async("playerctl", "previous")
```

#### Composante 2 : Boucle Média Temps Réel

**Dans main.py** : thread `_bt_media_loop()` qui tourne indépendamment toutes les 1 seconde.

```python
def _bt_media_loop(self):
    """Boucle 1s : fetch métadonnées BT, détecte connexion/déconnexion."""
    while self.state.get("running"):
        try:
            info = self.bt.get_current_track_info()
            is_now_connected = info["connected"]
            
            # Transition first connection
            if is_now_connected and not self._bt_was_connected:
                self._bt_was_connected = True
                self.gui.show_bt_card()  # Affiche carte BT
                self.gui.show_toast("✅ Téléphone connecté !")
            
            # Transition disconnect
            elif not is_now_connected and self._bt_was_connected:
                self._bt_was_connected = False
                self.gui.hide_bt_card()  # Masque carte BT
                self.gui.show_toast("❌ Déconnecté")
            
            # Update GUI si connecté
            if is_now_connected:
                self.gui.update_media_card(info)
        
        except Exception as e:
            logger.exception(f"_bt_media_loop error: {e}")
        
        time.sleep(1)
```

#### Composante 3 : Cover Art & Placeholder Vinyle

**Téléchargement Cover** :

```python
def _load_cover_art(art_url: str, cache_key: str) -> Optional[ctk.CTkImage]:
    """Télécharge https:// ou charge file://, redimensionne 200×200."""
    if not art_url or art_url == self._last_art_url:
        return self._cached_image
    
    img = None
    try:
        if art_url.startswith("file://"):
            img = Image.open(art_url[7:])
        elif art_url.startswith("http"):
            resp = urllib.request.urlopen(art_url, timeout=2)
            img = Image.open(io.BytesIO(resp.read()))
        
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        self._cached_image = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
        self._last_art_url = art_url
        return self._cached_image
    
    except Exception:
        # Fallback : placeholder vinyle généré PIL
        return self._placeholder_vinyl_image
```

**Placeholder Vinyle Généré** (src/gui.py, lignes 44–68) :

```python
def _make_placeholder_img(size: int = 200) -> Image.Image:
    """Disque vinyle stylisé couleurs EV-OS."""
    img = Image.new('RGB', (size, size), (27, 43, 59))  # CARD_CTK
    draw = ImageDraw.Draw(img)
    c = size // 2
    
    # Disque extérieur noir
    r_out = c - 4
    draw.ellipse([c - r_out, c - r_out, c + r_out, c + r_out], 
                 fill=(10, 17, 24))
    
    # Rainures vinyle (cercles concentriques)
    for r in range(r_out - 6, r_out - 36, -9):
        draw.ellipse([c - r, c - r, c + r, c + r], outline=(18, 28, 38), width=1)
    
    # Label central cyan
    r_lbl = c // 3
    draw.ellipse([c - r_lbl, c - r_lbl, c + r_lbl, c + r_lbl], fill=(13, 32, 44))
    draw.ellipse([c - r_lbl, c - r_lbl, c + r_lbl, c + r_lbl],
                 outline=(0, 229, 255), width=2)
    
    # Trou central cyan
    draw.ellipse([c - 7, c - 7, c + 7, c + 7], fill=(0, 229, 255))
    
    return img
```

**Transitions Visuelles** :

```python
# Première connexion BT
def show_bt_card(self):
    self.bt_card_frame.pack(before=self.nav_frame, fill='x', padx=10, pady=5)

# Déconnexion
def hide_bt_card(self):
    self.bt_card_frame.pack_forget()
```

#### Avantages de cette Architecture

✅ **Zéro dépendance cloud** — tout local BlueZ/PipeWire  
✅ **Compatible avec n'importe quel app audio** — Spotify, YouTube Music, Podcast, Local…  
✅ **AVRCP complet** — play/pause/next/prev depuis le Pi  
✅ **Cover art** — récupéré en local, pas de fetch API  
✅ **Autonome embarquée** — pas de compte Premium obligatoire  
✅ **Transitions fluides** — détection connexion/déconnexion en 1–2 sec  

---

## ⚙️ 3. Architecture & Fonctionnement Actuel

### Carte Globale des Modules

| Module | Rôle | Dépendances |
|--------|------|-------------|
| **main.py** | Orchestration app, boucles threads, reconnaissance facial temps réel | Tous les autres |
| **gui.py** | Interface CTkinter, layout responsive, clavier AZERTY, gestion cartes | customtkinter, PIL |
| **bluetooth_mgr.py** | Appairage BlueZ, métadonnées playerctl AVRCP | subprocess, threading |
| **hardware_comm.py** | Communication JSON série vers ESP32 (moteurs rétroviseur) | pyserial, json |
| **config.py** | Constantes : seuils détection, ports, modèles IA | os, logging |
| **detection.py** | Détection faciale ONNX (`det_10g.onnx`) | onnxruntime, numpy, cv2 |
| **embedding.py** | Embeddings ResNet 50 (`w600k_r50.onnx`) | onnxruntime, numpy, cv2 |
| **antispoofing.py** | Anti-spoofing PyTorch MiniFASNetV2 | torch |
| **recognition.py** | Reconnaissance : cosinus similarity, seuils | numpy, sklearn |
| **database.py** | Stockage embeddings NPZ (data/embeddings.npz) | numpy |
| **spotify_mgr.py** | (Hérité) Gestion Spotify legacy (non utilisé v0.1+) | spotipy |
| **weather_mgr.py** | Récupération météo locale | requests |
| **enroll_face.py** | Enrollment : capture multiple frames, moyenne embeddings | cv2, numpy |
| **utils.py** | Utilitaires : QC image, pose estimation, draw detections | cv2, numpy |
| **log_config.py** | Setup logging coloré structuré | logging, coloredlogs |

### Flux de Données (Diagramme ASCII)

```
┌────────────────────────────────────────────────────────────────────┐
│                    ENTREE UTILISATEUR (Pi5 Tactile)               │
│                      Écran 1280×800 / 1024×600                    │
│  [Accueil] [Véhicule] [Médias] [Paramètres]                       │
└─────────────┬────────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────────┐
│                   INTERFACE EV-OS (gui.py)                        │
│  ┌─ Sidebar (nav)      ┌─ Carte Média BT (nouveau)              │
│  │                      │   - Titre, Artiste, Cover             │
│  │ ┌─ Carte Accueil     │   - Boutons play/pause/next/prev      │
│  │ │ (reconnaissance    │   - Status connexion                   │
│  │ │  utilisateur)      │                                        │
│  │ │                    │ ┌─ Carte Véhicule                      │
│  │ │ ┌─ Clavier         │ │  - Position rétroviseur              │
│  │ │ │ Virtuel AZERTY   │ │  - Boutons ↑↓←→                      │
│  │ │ │ (prompt nom)     │ │  - Calibration zéro                  │
│  │ │                    │                                        │
│  │ └─ Carte Paramètres  └─ Clavier Virtuel (overlay)            │
│  │    - WiFi / BT Pair  └─ Toasts (notifications 3s)            │
│  └─────────────────────────────────────────────────────────────┘
└─────────────┬────────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────────┐
│                     BOUCLES THREADS DEMONS                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. VIDEO CAPTURE (main.py)                                 │ │
│  │    - Flux continu cv2.VideoCapture(0)                      │ │
│  │    - Détection facial ONNX toutes les 0.3s                 │ │
│  │    - Anti-spoofing check (image qualité)                   │ │
│  │    - Embedding cosinus → Reconnaissance                    │ │
│  │    - Update GUI avec frame annoté                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 2. BT MEDIA LOOP (main.py._bt_media_loop)                  │ │
│  │    - playerctl metadata (1s)                               │ │
│  │    - Détect connexion/déconnexion                          │ │
│  │    - Affiche/masque carte BT                               │ │
│  │    - Charge cover art (cache)                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 3. POLLING METEO (weather_mgr.py)                          │ │
│  │    - Récupère météo locale periodique                      │ │
│  │    - Update GUI                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────────┐
│              SORTIES MATERIELLES (Pi5 → Peripheriques)            │
│  ┌────────────────────┐                                           │
│  │ ESP32 (port série) │ ← JSON: {"cmd": "move_mirror", ...}     │
│  │ /dev/ttyUSB0       │                                           │
│  │ 115200 baud        │                                           │
│  └───────┬────────────┘                                           │
│          │                                                        │
│    ┌─────┴──────────────────────────────┐                        │
│    ↓                                    ↓                        │
│ ┌─────────────────────┐    ┌──────────────────────────┐          │
│ │ Moteurs Rétroviseur │    │ Périphériques Embarqués │          │
│ │ H/B/G/D / Calibrage │    │ (LEDs, buzzer, capteurs)│          │
│ └─────────────────────┘    └──────────────────────────┘          │
│                                                                  │
│  ┌────────────────────────┐                                     │
│  │ Audio Bluetooth Sink   │ ← PipeWire (espion de l'audio)     │
│  │ (BlueZ)                │   playerctl contrôle AVRCP         │
│  └────────────────────────┘                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Stack Technique Complet

**Backend (Reconnaissance & IA)** :
- **Vision** : OpenCV 4.11+
- **Détection/Embedding** : ONNX Runtime (CPU/GPU configurable)
- **Anti-spoofing** : PyTorch 2.7
- **Embeddings** : NumPy 2.2, scikit-learn 1.6 (cosinus)
- **Métadonnées** : PIL/Pillow 11.1

**Média & Système** :
- **Bluetooth A2DP** : BlueZ (service d'OS)
- **Contrôle AVRCP** : playerctl CLI (via subprocess)
- **Audio** : PipeWire daemon (service d'OS)
- **Série ESP32** : pyserial 3.5+

**Interface & Desktop** :
- **GUI Principal** : CustomTkinter 5.2.2
- **Cartes & MapView** : TkinterMapView 1.29+
- **Logging** : coloredlogs, logging standard Python

**Environnement** :
- **Python 3.11+** (officiellement testé sur Pi5)
- **Virtualenv** : `.venv` activé dans `start_cockpit.sh`
- **OS** : Raspberry Pi OS (Labwc/Wayland), Ubuntu Server (testable), macOS (dev/debug)

---

## 🚀 4. Installation & Démarrage Rapide

### Prérequis Système (Raspberry Pi 5)

```bash
# Mise à jour système
sudo apt update && sudo apt upgrade -y

# Dépendances système obligatoires
sudo apt install -y \
  python3.11 python3.11-venv python3.11-dev \
  libopenblas0 libblas3 liblapack3 gfortran \
  libatlas-base-dev \
  libjasper-dev libtiff-dev libjasper1 \
  libharfbuzz0b libwebp6 \
  libatlas3-base \
  libjasper1 \
  libharfbuzz0b \
  libwebp6 \
  libopenjp2-7 \
  libtiff6 \
  libopenblas-dev \
  bluez bluez-tools playerctl \
  pipewire pipewire-media-session \
  libpipewire-0.3-dev \
  git cmake

# Si vous utilisez GPIO / capteurs supplémentaires
sudo apt install -y raspi-config

# Service Bluetooth & PipeWire (activation)
sudo systemctl enable bluetooth
sudo systemctl start bluetooth
sudo systemctl --user enable pipewire
systemctl --user start pipewire
```

### Clonage & Setup Python

```bash
# Clone du projet
cd ~
git clone https://github.com/yourusername/projetmr-gomes.git
cd projetmr-gomes

# Créer virtualenv Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Installer dépendances Python
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Créer dossier modèles IA
mkdir -p models
mkdir -p data

# Télécharger modèles ONNX (si pas présents)
# det_10g.onnx, w600k_r50.onnx dans models/
# (Voir IMPLEMENTATION_GUIDE.md pour liens)
```

### Configuration (config.py)

Éditer `/src/config.py` pour votre environnement :

```python
# === Port série ESP32 ===
SERIAL_PORT = '/dev/ttyUSB0'         # Vérifier avec: ls /dev/tty*
SERIAL_BAUDRATE = 115200

# === Seuils reconnaissance faciale ===
RECOGNITION_THRESHOLD = 0.50         # Cosinus similarity min
ADAPTATION_THRESHOLD = 0.65           # Pour apprentissage continu
BLUR_THRESHOLD = 75.0                 # QC image (Laplacian variance)

# === Modèles IA ===
# Vérifier que ces fichiers existent dans models/
DETECTOR_PATH = os.path.join("models", "det_10g.onnx")
EMBEDDER_PATH = os.path.join("models", "w600k_r50.onnx")
ANTI_SPOOFING_PATH = os.path.join("models", "2.7_80x80_MiniFASNetV2.pth")

# === GUI ===
GUI_WINDOW_TITLE = "EV-OS Mehari Cockpit"
GUI_UPDATE_INTERVAL_MS = 33           # ~30 FPS
```

### Démarrage Application

#### Mode Développement (Fenêtrée)

```bash
cd /Users/charlesaugustendiaye/projetmr-gomes
source .venv/bin/activate
python -m src.main
```

Escape pour quitter fullscreen et revenir à fenêtrage normal.

#### Mode Production (Kiosk Fullscreen Pi5)

```bash
# Exécuter via start_cockpit.sh
cd /Users/charlesaugustendiaye/projetmr-gomes
./start_cockpit.sh

# Ou directement
source .venv/bin/activate
DISPLAY=:0 python -m src.main
```

#### Autostart au Boot (Systemd User Service)

Créer `~/.config/systemd/user/cockpit.service` :

```ini
[Unit]
Description=EV-OS Cockpit Mehari Dashboard
After=multi-user.target
Wants=bluetooth.service

[Service]
Type=simple
Environment="DISPLAY=:0"
Environment="XAUTHORITY=%h/.Xauthority"
ExecStart=/home/pi/projetmr-gomes/start_cockpit.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

```bash
# Activer service
systemctl --user daemon-reload
systemctl --user enable cockpit
systemctl --user start cockpit

# Voir logs en direct
journalctl --user -u cockpit -f
```

### Configuration Bluetooth (BlueZ + PipeWire)

**Renommer le Pi** (optionnel, en tant que "mehari") :

```bash
sudo hostnamectl set-hostname mehari
sudo systemctl restart avahi-daemon  # Si utilisé

# Vérifier
hostnamectl
# Static hostname: mehari
```

**Vérifier BlueZ & PipeWire** :

```bash
# BlueZ
systemctl status bluetooth
bluetoothctl --version

# PipeWire
systemctl --user status pipewire
pactl info  # (via PipeWire)

# playerctl
playerctl --version
```

**Test manuel appairage** :

```bash
# Depuis Pi5
bluetoothctl

# Dans le prompt bluetoothctl
> power on
> agent NoInputNoOutput
> default-agent
> discoverable on
> pairable on

# Depuis téléphone : chercher "mehari" en Bluetooth disponibles
# Appairer et connecter A2DP

# Sur Pi5, vérifier connexion
> devices
[...] MAC ADDRESS LE Public Téléphone_Name
# ← "Téléphone_Name" doit être "connected"
```

**Test playerctl** :

```bash
# Lancer musique sur téléphone (via Spotify, YouTube Music, etc.)
playerctl status
# Playing

playerctl metadata title
# "Nom de la chanson"

playerctl metadata artist
# "Nom de l'artiste"

playerctl play-pause
# Bascule lecture/pause sur le téléphone
```

### Vérifications Post-Démarrage

```bash
# 1. Reconnaissance faciale en direct
python -m src.main
# → Webcam s'affiche (si connectée), détection FPS affichées

# 2. Clavier virtuel
# Cliquer sur un champ texte → clavier AZERTY doit apparaître en bas

# 3. Bluetooth
# Appuyer sur "Associer téléphone" → Pi devient découvrable
# Connecter le téléphone → carte BT doit s'afficher dans l'interface

# 4. ESP32 (rétroviseur)
# Appuyer sur boutons ↑/↓/←/→ → commandes JSON vers ESP32 via /dev/ttyUSB0
# Vérifier logs : `tail -f .venv/*/site-packages/../app.log` si LOG_TO_FILE=True
```

### Raccourcis Clavier Développement

| Touche | Action |
|--------|--------|
| **Escape** | Quitter fullscreen (redevient fenêtrée) |
| **Q** (en mode enroll) | Quitter enrollment |
| **S** (en mode enroll) | Sauvegarder embedding |

---

## 📚 Documentation Complémentaire

- **IMPLEMENTATION_GUIDE.md** : Setup détaillé, modèles IA, troubleshooting
- **README.md** : Vue générale du projet, contributions
- **src/config.py** : Toutes les constantes configurables
- **src/log_config.py** : Logging structuré coloré

---

## 🔧 Troubleshooting Rapide

### "Erreur `/dev/ttyUSB0` not found"
- Vérifier ESP32 connecté : `ls -la /dev/tty*`
- Installer driver CH340/CP210x : `sudo apt install ch340-driver`
- Vérifier permissions : `sudo usermod -a -G dialout $USER` (reboot requis)

### "playerctl : command not found"
- Installer playerctl : `sudo apt install playerctl`
- Vérifier PipeWire : `systemctl --user status pipewire`

### "bluetoothctl : command not found"
- Installer BlueZ : `sudo apt install bluez bluez-tools`
- Service : `sudo systemctl enable bluetooth && sudo systemctl start bluetooth`

### "CustomTkinter / PIL import error"
- Réinstaller dépendances : `pip install -r requirements.txt --force-reinstall`
- Vérifier venv actif : `which python` doit montrer `.venv/bin/python`

### "Écran tactile ne répond pas / clavier virtuel décalé"
- Calibrer écran tactile (Labwc) : vérifier `/etc/libinput/calibration.conf`
- Redémarrer composeur Wayland : `pkill labwc`
- Sur macOS dev, écran peut être différent : layout ajustable dans `gui.py`

---

**Dernière mise à jour** : 2026-06-19  
**Version EV-OS** : v0.1.1 (Reconnaissance Facial code propre)  
**Auteur** : Charles Ndiaye (ndiaye.charles@proton.me)  
**License** : [À définir]

---
# 🚗 EV-OS: Mehari Cockpit (English Version)

**EV-OS** is an embedded touchscreen dashboard designed for an electric Mehari. This ecosystem combines facial recognition, smart mirror motor control, Bluetooth A2DP Sink media access, and real-time weather on a Raspberry Pi 5 with a 10.1" touchscreen. The project embodies a philosophy of technical autonomy: zero dependency on paid cloud APIs (Spotify Web API), full local control, and a 100% Python non-blocking interface via CustomTkinter.

**Project character**: pragmatic embedded engineering. No heavy framework, just efficient Python + BlueZ/PipeWire for Bluetooth, OpenCV/ONNX for vision, and a modern Tkinter interface designed for touch.

---

## 🏗️ 1. Project Origin & Starting Base

### Initial Context

The project began as a Python facial recognition demo on a Raspberry Pi, with a basic Tkinter interface. Three components formed the backbone:

1. **Face detection**: OpenCV + ONNX Runtime (`det_10g.onnx`), converting images into high-dimensional embeddings via ResNet 50 (`w600k_r50.onnx`)
2. **Anti-spoofing**: PyTorch MiniFASNetV2 (`2.7_80x80_MiniFASNetV2.pth`) to reject photo/video attacks
3. **Recognition**: cosine comparison of embeddings with a configurable threshold (0.50 by default)

### Initial Spotify API Architecture (Limitation)

v0.0.1 integrated the **Spotify Web API** for embedded music:
- ✅ Metadata: title, artist, cover art
- ⚠️ **Critical issues**:
  - OAuth 2.0 authentication + refresh token required → external infrastructure
  - Premium account mandatory for the API (cost)
  - No local audio access → dependency on cloud streaming
  - Fragile in an embedded environment with intermittent connectivity

**Verdict**: unsuitable for a standalone electric Mehari.

### Deep Learning Stack (Unchanged)

```python
# Models used (config.py)
DETECTOR_MODEL = "det_10g.onnx"          # Face detection
EMBEDDER_MODEL = "w600k_r50.onnx"        # 512-D embeddings
ANTI_SPOOFING_MODEL = "2.7_80x80_MiniFASNetV2.pth"  # Liveness check
```

**Stack**: NumPy, OpenCV, ONNX Runtime (CPU/GPU), PyTorch, scikit-learn.

---

## 🛠️ 2. Improvements & Changes Made

### ✨ CustomTkinter Migration & EV-OS Design System

**Before**: plain Tkinter (default gray/blue, rigid layout).

**After v0.1.0**: complete redesign in **CustomTkinter 5.2+** with a proprietary dark theme.

#### EV-OS Color Palette (Cyberpunk + Pi5)

```python
# src/gui.py — Palette definition
BG       = '#0A1118'       # Ultra-deep black background
SIDEBAR  = '#0D141C'       # Slightly lighter sidebar
CARD     = '#16222F'       # Cards / main panels
CARD_CTK = '#1B2B3B'       # CTkFrame cards (depth)
ENTRY_BG = '#0F1C28'       # Input fields / listbox
TEXT     = '#FFFFFF'       # Main text (white)
SUB      = '#94A3B8'       # Secondary text (blue-gray)
CYAN     = '#00E5FF'       # EV-OS accent (electric cyan)
NAV_ACT  = '#1A3A4A'       # Active nav button
BLACK    = '#000000'       # For shadows/contrast
```

All components (CTkFrame, CTkButton, CTkLabel, CTkProgressBar, CTkEntry) use this palette.
**Result**: a consistent, modern interface, readable on a 10.1" touchscreen even in full sunlight.

#### Non-Blocking Layout

- All network, AI, and system calls run on **daemon threads** independently
- Responsive GUI even during facial recognition or Bluetooth connection
- `threading.Thread(..., daemon=True)` used throughout
- **Main Tkinter loop** unchanged — simply driven by `update()` via callbacks

### 🎮 Virtual AZERTY Touch Keyboard – CTkVirtualKeyboard

**Problem solved**: Wayland + XWayland on Raspberry Pi 5 = conflicts with onboard/squeekboard (crashes, unresponsive).

**Solution**: 100% Python, fully self-contained CTkVirtualKeyboard implementation (`src/gui.py`, lines 71–186).

#### Architecture

```python
class CTkVirtualKeyboard(ctk.CTkFrame):
    """Standalone AZERTY CTkinter keyboard, zero system dependency."""
    
    def show(self, target: ctk.CTkEntry, on_validate=None):
        """Displays the keyboard anchored at the bottom, targeting the entry."""
        self._target = target
        self._on_validate_cb = on_validate
        self.place(relx=0, rely=1.0, anchor='sw', relwidth=1.0)  # sticky bottom
        self.lift()
    
    def hide(self):
        self.place_forget()
    
    def _press(self, char: str):
        """Inserts char into the target CTkEntry."""
        inner = self._target._entry  # underlying tk.Entry
        inner.insert('insert', char)
    
    def _backspace(self):
        """Deletes the character before the cursor."""
        inner = self._target._entry
        pos = inner.index('insert')
        if pos > 0:
            inner.delete(pos - 1, pos)
```

#### AZERTY Layout

```python
_AZERTY = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['A', 'Z', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['Q', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M'],
    ['W', 'X', 'C', 'V', 'B', 'N', '@', '.', '_', '-'],
]
```

Each key is a `CTkButton` 48px tall. Bottom row: **Space**, **⌫ (red backspace)**, **✓ OK (cyan)**. Pressing **OK** calls `on_validate_cb()` or hides the keyboard.

**Usage**:

```python
# Driver name input with overlay prompt
def prompt_driver_name():
    dialog = ctk.CTkToplevel(self.root)
    entry = ctk.CTkEntry(dialog, ...)
    self._vkb.show(entry, on_validate=lambda: save_driver(entry.get()))
    self.root.wait_variable(...)  # Waits for validation
```

### 🔧 Raspberry Pi 5 Adaptation & Production Kiosk

**Target environment**:
- Raspberry Pi 5 (Pi4 possible, but slower)
- Labwc (lightweight Wayland compositor) + 1280×800 or 1024×600 touchscreen
- USB serial port → `/dev/ttyUSB0` for the mirror ESP32

#### ESP32 Serial Configuration (hardware_comm.py)

```python
SERIAL_PORT = '/dev/ttyUSB0'  # Defined in config.py
SERIAL_BAUDRATE = 115200      # ESP32 standard
```

Bidirectional JSON communication:
```json
{
  "cmd": "move_mirror",
  "position": "up",
  "duration_ms": 500
}
```

#### Kiosk & Fullscreen Mode

```python
# src/gui.py
class FaceRecognitionGUI:
    def __init__(self, root: ctk.CTk):
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
```

The kiosk starts in fullscreen. **Escape** toggles fullscreen (development). In production, Escape is locked via udev/systemd.

#### Startup Script (start_cockpit.sh)

```bash
#!/bin/bash
source .venv/bin/activate
DISPLAY=:0 python -m src.main
```

Runs at boot via user crontab or a systemd user service:
```bash
# ~/.config/systemd/user/cockpit.service (optional)
[Unit]
Description=EV-OS Cockpit Dashboard
After=multi-user.target

[Service]
Type=simple
ExecStart=/home/pi/projetmr-gomes/start_cockpit.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

### 🎵 A2DP Sink Bluetooth Media System – Audio Revolution

**v0.0.1 problem**: paid Spotify API, no local audio autonomy.

**New Architecture (v0.1.0+)**:

```
Phone (Spotify, YouTube Music, etc.)
    ↓ A2DP Stream
Raspberry Pi (BlueZ Sink)
    ↓ PipeWire (audio daemon)
    ↓ playerctl (AVRCP control + metadata)
EV-OS GUI
    ↓ Displays cover art, title, artist
    ↓ Play/pause/next/prev buttons
```

#### Component 1: BluetoothManager (src/bluetooth_mgr.py)

Standalone class managing pairing + metadata.

**Discoverable Pairing**:

```python
def make_discoverable(self, on_done=None):
    """Makes the Pi discoverable + pairable in the background."""
    cmds = (
        "power on\n"
        "agent NoInputNoOutput\n"
        "default-agent\n"
        "discoverable on\n"
        "pairable on\n"
    )
    # Pipe `cmds` → bluetoothctl stdin (daemon thread)
    # Calls on_done(status_str) once done
```

The Pi becomes visible under its hostname (e.g. "mehari") for 3 minutes. The user taps "Pair a phone" and stays within Bluetooth range (3–10m depending on Pi5 + antenna).

**Reading AVRCP Metadata**:

```python
def get_current_track_info(self) -> dict:
    """Returns title/artist/state + cover via playerctl."""
    # playerctl status → "Playing" | "Paused" | error
    # playerctl metadata title
    # playerctl metadata artist
    # playerctl metadata mpris:artUrl  ← file:// or https://
    return {
        "connected":  bool,
        "title":      str,
        "artist":     str,
        "is_playing": bool,
        "art_url":    str,
    }
```

**AVRCP Commands** (Fire-and-Forget):

```python
def toggle_play_pause(self):
    self._run_async("playerctl", "play-pause")

def next_track(self):
    self._run_async("playerctl", "next")

def prev_track(self):
    self._run_async("playerctl", "previous")
```

#### Component 2: Real-Time Media Loop

**In main.py**: a `_bt_media_loop()` thread that runs independently every 1 second.

```python
def _bt_media_loop(self):
    """1s loop: fetch BT metadata, detect connection/disconnection."""
    while self.state.get("running"):
        try:
            info = self.bt.get_current_track_info()
            is_now_connected = info["connected"]
            
            # First connection transition
            if is_now_connected and not self._bt_was_connected:
                self._bt_was_connected = True
                self.gui.show_bt_card()  # Show BT card
                self.gui.show_toast("✅ Phone connected!")
            
            # Disconnect transition
            elif not is_now_connected and self._bt_was_connected:
                self._bt_was_connected = False
                self.gui.hide_bt_card()  # Hide BT card
                self.gui.show_toast("❌ Disconnected")
            
            # Update GUI if connected
            if is_now_connected:
                self.gui.update_media_card(info)
        
        except Exception as e:
            logger.exception(f"_bt_media_loop error: {e}")
        
        time.sleep(1)
```

#### Component 3: Cover Art & Vinyl Placeholder

**Cover Download**:

```python
def _load_cover_art(art_url: str, cache_key: str) -> Optional[ctk.CTkImage]:
    """Downloads https:// or loads file://, resizes to 200×200."""
    if not art_url or art_url == self._last_art_url:
        return self._cached_image
    
    img = None
    try:
        if art_url.startswith("file://"):
            img = Image.open(art_url[7:])
        elif art_url.startswith("http"):
            resp = urllib.request.urlopen(art_url, timeout=2)
            img = Image.open(io.BytesIO(resp.read()))
        
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        self._cached_image = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
        self._last_art_url = art_url
        return self._cached_image
    
    except Exception:
        # Fallback: PIL-generated vinyl placeholder
        return self._placeholder_vinyl_image
```

**Generated Vinyl Placeholder** (src/gui.py, lines 44–68):

```python
def _make_placeholder_img(size: int = 200) -> Image.Image:
    """Stylized vinyl record in EV-OS colors."""
    img = Image.new('RGB', (size, size), (27, 43, 59))  # CARD_CTK
    draw = ImageDraw.Draw(img)
    c = size // 2
    
    # Outer black disc
    r_out = c - 4
    draw.ellipse([c - r_out, c - r_out, c + r_out, c + r_out], 
                 fill=(10, 17, 24))
    
    # Vinyl grooves (concentric circles)
    for r in range(r_out - 6, r_out - 36, -9):
        draw.ellipse([c - r, c - r, c + r, c + r], outline=(18, 28, 38), width=1)
    
    # Central cyan label
    r_lbl = c // 3
    draw.ellipse([c - r_lbl, c - r_lbl, c + r_lbl, c + r_lbl], fill=(13, 32, 44))
    draw.ellipse([c - r_lbl, c - r_lbl, c + r_lbl, c + r_lbl],
                 outline=(0, 229, 255), width=2)
    
    # Central cyan hole
    draw.ellipse([c - 7, c - 7, c + 7, c + 7], fill=(0, 229, 255))
    
    return img
```

**Visual Transitions**:

```python
# First BT connection
def show_bt_card(self):
    self.bt_card_frame.pack(before=self.nav_frame, fill='x', padx=10, pady=5)

# Disconnection
def hide_bt_card(self):
    self.bt_card_frame.pack_forget()
```

#### Benefits of this Architecture

✅ **Zero cloud dependency** — everything local via BlueZ/PipeWire  
✅ **Compatible with any audio app** — Spotify, YouTube Music, Podcasts, Local files…  
✅ **Full AVRCP support** — play/pause/next/prev from the Pi  
✅ **Cover art** — fetched locally, no API call needed  
✅ **Standalone autonomy** — no mandatory Premium account  
✅ **Smooth transitions** — connection/disconnection detected in 1–2 sec  

---

## ⚙️ 3. Current Architecture & Operation

### Global Module Map

| Module | Role | Dependencies |
|--------|------|-------------|
| **main.py** | App orchestration, thread loops, real-time facial recognition | All other modules |
| **gui.py** | CTkinter interface, responsive layout, AZERTY keyboard, card management | customtkinter, PIL |
| **bluetooth_mgr.py** | BlueZ pairing, playerctl AVRCP metadata | subprocess, threading |
| **hardware_comm.py** | JSON serial communication with ESP32 (mirror motors) | pyserial, json |
| **config.py** | Constants: detection thresholds, ports, AI models | os, logging |
| **detection.py** | ONNX facial detection (`det_10g.onnx`) | onnxruntime, numpy, cv2 |
| **embedding.py** | ResNet 50 embeddings (`w600k_r50.onnx`) | onnxruntime, numpy, cv2 |
| **antispoofing.py** | PyTorch MiniFASNetV2 anti-spoofing | torch |
| **recognition.py** | Recognition: cosine similarity, thresholds | numpy, sklearn |
| **database.py** | NPZ embedding storage (data/embeddings.npz) | numpy |
| **spotify_mgr.py** | (Legacy) Spotify management (unused in v0.1+) | spotipy |
| **weather_mgr.py** | Local weather retrieval | requests |
| **enroll_face.py** | Enrollment: multi-frame capture, embedding averaging | cv2, numpy |
| **utils.py** | Utilities: image QC, pose estimation, draw detections | cv2, numpy |
| **log_config.py** | Structured colored logging setup | logging, coloredlogs |

### Data Flow (ASCII Diagram)

```
┌────────────────────────────────────────────────────────────────────┐
│                    USER INPUT (Pi5 Touchscreen)                   │
│                      1280×800 / 1024×600 Screen                   │
│  [Home] [Vehicle] [Media] [Settings]                              │
└─────────────┬────────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────────┐
│                   EV-OS INTERFACE (gui.py)                        │
│  ┌─ Sidebar (nav)      ┌─ BT Media Card (new)                    │
│  │                      │   - Title, Artist, Cover               │
│  │ ┌─ Home Card         │   - Play/pause/next/prev buttons        │
│  │ │ (user              │   - Connection status                   │
│  │ │  recognition)      │                                        │
│  │ │                    │ ┌─ Vehicle Card                        │
│  │ │ ┌─ Virtual         │ │  - Mirror position                    │
│  │ │ │ AZERTY Keyboard  │ │  - ↑↓←→ buttons                       │
│  │ │ │ (name prompt)    │ │  - Zero calibration                  │
│  │ │                    │                                        │
│  │ └─ Settings Card     └─ Virtual Keyboard (overlay)            │
│  │    - WiFi / BT Pair  └─ Toasts (3s notifications)             │
│  └─────────────────────────────────────────────────────────────┘
└─────────────┬────────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────────┐
│                     DAEMON THREAD LOOPS                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. VIDEO CAPTURE (main.py)                                 │ │
│  │    - Continuous cv2.VideoCapture(0) stream                 │ │
│  │    - ONNX facial detection every 0.3s                      │ │
│  │    - Anti-spoofing check (image quality)                   │ │
│  │    - Cosine embedding → Recognition                        │ │
│  │    - Update GUI with annotated frame                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 2. BT MEDIA LOOP (main.py._bt_media_loop)                  │ │
│  │    - playerctl metadata (1s)                               │ │
│  │    - Detect connection/disconnection                       │ │
│  │    - Show/hide BT card                                     │ │
│  │    - Load cover art (cached)                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 3. WEATHER POLLING (weather_mgr.py)                        │ │
│  │    - Periodic local weather retrieval                      │ │
│  │    - Update GUI                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────────┐
│              HARDWARE OUTPUTS (Pi5 → Peripherals)                 │
│  ┌────────────────────┐                                           │
│  │ ESP32 (serial port)│ ← JSON: {"cmd": "move_mirror", ...}     │
│  │ /dev/ttyUSB0       │                                           │
│  │ 115200 baud        │                                           │
│  └───────┬────────────┘                                           │
│          │                                                        │
│    ┌─────┴──────────────────────────────┐                        │
│    ↓                                    ↓                        │
│ ┌─────────────────────┐    ┌──────────────────────────┐          │
│ │ Mirror Motors        │    │ Embedded Peripherals     │          │
│ │ Up/Down/Left/Right /│    │ (LEDs, buzzer, sensors)  │          │
│ │ Calibration          │    │                           │          │
│ └─────────────────────┘    └──────────────────────────┘          │
│                                                                  │
│  ┌────────────────────────┐                                     │
│  │ Bluetooth Audio Sink   │ ← PipeWire (audio sink)            │
│  │ (BlueZ)                │   playerctl AVRCP control          │
│  └────────────────────────┘                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Full Tech Stack

**Backend (Recognition & AI)**:
- **Vision**: OpenCV 4.11+
- **Detection/Embedding**: ONNX Runtime (CPU/GPU configurable)
- **Anti-spoofing**: PyTorch 2.7
- **Embeddings**: NumPy 2.2, scikit-learn 1.6 (cosine)
- **Metadata**: PIL/Pillow 11.1

**Media & System**:
- **Bluetooth A2DP**: BlueZ (OS service)
- **AVRCP Control**: playerctl CLI (via subprocess)
- **Audio**: PipeWire daemon (OS service)
- **ESP32 Serial**: pyserial 3.5+

**Interface & Desktop**:
- **Main GUI**: CustomTkinter 5.2.2
- **Maps & MapView**: TkinterMapView 1.29+
- **Logging**: coloredlogs, standard Python logging

**Environment**:
- **Python 3.11+** (officially tested on Pi5)
- **Virtualenv**: `.venv` activated in `start_cockpit.sh`
- **OS**: Raspberry Pi OS (Labwc/Wayland), Ubuntu Server (testable), macOS (dev/debug)

---

## 🚀 4. Installation & Quick Start

### System Prerequisites (Raspberry Pi 5)

```bash
# System update
sudo apt update && sudo apt upgrade -y

# Required system dependencies
sudo apt install -y \
  python3.11 python3.11-venv python3.11-dev \
  libopenblas0 libblas3 liblapack3 gfortran \
  libatlas-base-dev \
  libjasper-dev libtiff-dev libjasper1 \
  libharfbuzz0b libwebp6 \
  libatlas3-base \
  libjasper1 \
  libharfbuzz0b \
  libwebp6 \
  libopenjp2-7 \
  libtiff6 \
  libopenblas-dev \
  bluez bluez-tools playerctl \
  pipewire pipewire-media-session \
  libpipewire-0.3-dev \
  git cmake

# If using GPIO / additional sensors
sudo apt install -y raspi-config

# Bluetooth & PipeWire services (activation)
sudo systemctl enable bluetooth
sudo systemctl start bluetooth
sudo systemctl --user enable pipewire
systemctl --user start pipewire
```

### Cloning & Python Setup

```bash
# Clone the project
cd ~
git clone https://github.com/yourusername/projetmr-gomes.git
cd projetmr-gomes

# Create a Python 3.11 virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create model folders
mkdir -p models
mkdir -p data

# Download ONNX models (if not already present)
# det_10g.onnx, w600k_r50.onnx into models/
# (See IMPLEMENTATION_GUIDE.md for links)
```

### Configuration (config.py)

Edit `/src/config.py` for your environment:

```python
# === ESP32 Serial Port ===
SERIAL_PORT = '/dev/ttyUSB0'         # Check with: ls /dev/tty*
SERIAL_BAUDRATE = 115200

# === Facial Recognition Thresholds ===
RECOGNITION_THRESHOLD = 0.50         # Min cosine similarity
ADAPTATION_THRESHOLD = 0.65           # For continuous learning
BLUR_THRESHOLD = 75.0                 # Image QC (Laplacian variance)

# === AI Models ===
# Check that these files exist in models/
DETECTOR_PATH = os.path.join("models", "det_10g.onnx")
EMBEDDER_PATH = os.path.join("models", "w600k_r50.onnx")
ANTI_SPOOFING_PATH = os.path.join("models", "2.7_80x80_MiniFASNetV2.pth")

# === GUI ===
GUI_WINDOW_TITLE = "EV-OS Mehari Cockpit"
GUI_UPDATE_INTERVAL_MS = 33           # ~30 FPS
```

### Starting the Application

#### Development Mode (Windowed)

```bash
cd /Users/charlesaugustendiaye/projetmr-gomes
source .venv/bin/activate
python -m src.main
```

Press Escape to exit fullscreen and return to windowed mode.

#### Production Mode (Pi5 Kiosk Fullscreen)

```bash
# Run via start_cockpit.sh
cd /Users/charlesaugustendiaye/projetmr-gomes
./start_cockpit.sh

# Or directly
source .venv/bin/activate
DISPLAY=:0 python -m src.main
```

#### Autostart on Boot (Systemd User Service)

Create `~/.config/systemd/user/cockpit.service`:

```ini
[Unit]
Description=EV-OS Cockpit Mehari Dashboard
After=multi-user.target
Wants=bluetooth.service

[Service]
Type=simple
Environment="DISPLAY=:0"
Environment="XAUTHORITY=%h/.Xauthority"
ExecStart=/home/pi/projetmr-gomes/start_cockpit.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

```bash
# Enable the service
systemctl --user daemon-reload
systemctl --user enable cockpit
systemctl --user start cockpit

# View live logs
journalctl --user -u cockpit -f
```

### Bluetooth Configuration (BlueZ + PipeWire)

**Rename the Pi** (optional, as "mehari"):

```bash
sudo hostnamectl set-hostname mehari
sudo systemctl restart avahi-daemon  # If used

# Verify
hostnamectl
# Static hostname: mehari
```

**Check BlueZ & PipeWire**:

```bash
# BlueZ
systemctl status bluetooth
bluetoothctl --version

# PipeWire
systemctl --user status pipewire
pactl info  # (via PipeWire)

# playerctl
playerctl --version
```

**Manual Pairing Test**:

```bash
# From the Pi5
bluetoothctl

# At the bluetoothctl prompt
> power on
> agent NoInputNoOutput
> default-agent
> discoverable on
> pairable on

# From the phone: look for "mehari" in available Bluetooth devices
# Pair and connect A2DP

# On the Pi5, verify the connection
> devices
[...] MAC ADDRESS LE Public Phone_Name
# ← "Phone_Name" must show as "connected"
```

**playerctl Test**:

```bash
# Play music on the phone (via Spotify, YouTube Music, etc.)
playerctl status
# Playing

playerctl metadata title
# "Song name"

playerctl metadata artist
# "Artist name"

playerctl play-pause
# Toggles play/pause on the phone
```

### Post-Startup Checks

```bash
# 1. Live facial recognition
python -m src.main
# → Webcam displays (if connected), detection FPS shown

# 2. Virtual keyboard
# Tap a text field → AZERTY keyboard should appear at the bottom

# 3. Bluetooth
# Tap "Pair a phone" → Pi becomes discoverable
# Connect the phone → BT card should appear in the interface

# 4. ESP32 (mirror)
# Tap the ↑/↓/←/→ buttons → JSON commands sent to ESP32 via /dev/ttyUSB0
# Check logs: `tail -f .venv/*/site-packages/../app.log` if LOG_TO_FILE=True
```

### Development Keyboard Shortcuts

| Key | Action |
|--------|--------|
| **Escape** | Exit fullscreen (back to windowed) |
| **Q** (in enroll mode) | Quit enrollment |
| **S** (in enroll mode) | Save embedding |

---

## 📚 Additional Documentation

- **IMPLEMENTATION_GUIDE.md**: Detailed setup, AI models, troubleshooting
- **README.md**: General project overview, contributions
- **src/config.py**: All configurable constants
- **src/log_config.py**: Structured colored logging

---

## 🔧 Quick Troubleshooting

### "`/dev/ttyUSB0` not found" error
- Check that the ESP32 is connected: `ls -la /dev/tty*`
- Install the CH340/CP210x driver: `sudo apt install ch340-driver`
- Check permissions: `sudo usermod -a -G dialout $USER` (reboot required)

### "playerctl: command not found"
- Install playerctl: `sudo apt install playerctl`
- Check PipeWire: `systemctl --user status pipewire`

### "bluetoothctl: command not found"
- Install BlueZ: `sudo apt install bluez bluez-tools`
- Service: `sudo systemctl enable bluetooth && sudo systemctl start bluetooth`

### "CustomTkinter / PIL import error"
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Check active venv: `which python` should show `.venv/bin/python`

### "Touchscreen not responding / virtual keyboard misaligned"
- Calibrate the touchscreen (Labwc): check `/etc/libinput/calibration.conf`
- Restart the Wayland compositor: `pkill labwc`
- On macOS dev, the screen may differ: layout adjustable in `gui.py`

---

**Last updated**: 2026-06-19  
**EV-OS Version**: v0.1.1 (Clean Facial Recognition code)  
**Author**: Charles Ndiaye (ndiaye.charles@proton.me)  
**License**: [To be defined]****
