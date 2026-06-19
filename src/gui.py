# src/gui.py
import tkinter as tk
from tkinter import ttk
import io as _io
import urllib.request
from PIL import Image, ImageDraw, ImageTk
from datetime import datetime
import cv2
import numpy as np
import logging
import os
from typing import Optional
from src import config

try:
    from tkintermapview import TkinterMapView
    import geocoder as _geocoder
    _MAP_AVAILABLE = True
except ImportError:
    _MAP_AVAILABLE = False
    _geocoder = None

import customtkinter as ctk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

logger = logging.getLogger(__name__)

# ── Palette EV-OS ─────────────────────────────────────────────────────────────
BG       = '#0A1118'
SIDEBAR  = '#0D141C'
CARD     = '#16222F'
CARD_CTK = '#1B2B3B'   # cartes CTkFrame (légèrement plus clair pour profondeur)
ENTRY_BG = '#0F1C28'   # champs de saisie / listbox
TEXT     = '#FFFFFF'
SUB      = '#94A3B8'   # gris bleuté discret
CYAN     = '#00E5FF'
NAV_ACT  = '#1A3A4A'
BLACK    = '#000000'


# ── Pochette par défaut (vinyle sombre / accent cyan) ─────────────────────────

def _make_placeholder_img(size: int = 200) -> Image.Image:
    """Génère un disque vinyle stylisé aux couleurs du thème EV-OS."""
    img  = Image.new('RGB', (size, size), (27, 43, 59))   # CARD_CTK
    draw = ImageDraw.Draw(img)
    c    = size // 2

    # Disque extérieur noir
    r_out = c - 4
    draw.ellipse([c - r_out, c - r_out, c + r_out, c + r_out], fill=(10, 17, 24))

    # Rainures vinyle (cercles concentriques discrets)
    for r in range(r_out - 6, r_out - 36, -9):
        draw.ellipse([c - r, c - r, c + r, c + r], outline=(18, 28, 38), width=1)

    # Label central coloré
    r_lbl = c // 3
    draw.ellipse([c - r_lbl, c - r_lbl, c + r_lbl, c + r_lbl], fill=(13, 32, 44))
    draw.ellipse([c - r_lbl, c - r_lbl, c + r_lbl, c + r_lbl],
                 outline=(0, 229, 255), width=2)

    # Trou central cyan
    r_h = 7
    draw.ellipse([c - r_h, c - r_h, c + r_h, c + r_h], fill=(0, 229, 255))

    return img


# ── Clavier virtuel CustomTkinter (pur Python, zéro dépendance système) ──────

_AZERTY = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['A', 'Z', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['Q', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M'],
    ['W', 'X', 'C', 'V', 'B', 'N', '@', '.', '_', '-'],
]

_KB_H      = 48   # hauteur d'une touche (px)
_KB_BTN_BG = '#1B2B3B'
_KB_HOV    = '#1A3A4A'


class CTkVirtualKeyboard(ctk.CTkFrame):
    """Clavier AZERTY CustomTkinter autonome.

    Usage :
        kb.show(entry_widget)            # affiche + cible l'entrée
        kb.show(entry, on_validate=fn)   # fn() appelée quand ✓ OK est pressé
        kb.hide()                        # masque
    """

    def __init__(self, master):
        super().__init__(master, fg_color=SIDEBAR, corner_radius=0)
        self._target: Optional[ctk.CTkEntry] = None
        self._on_validate_cb = None
        self._build()

    # ── Construction des touches ──────────────────────────────────────────────
    def _build(self):
        _kw = dict(
            height=_KB_H, corner_radius=8,
            fg_color=_KB_BTN_BG, hover_color=_KB_HOV,
            text_color=TEXT,
            font=ctk.CTkFont(family='Helvetica', size=15, weight='bold'),
        )

        for row_chars in _AZERTY:
            row_frame = ctk.CTkFrame(self, fg_color='transparent')
            row_frame.pack(fill='x', padx=6, pady=3)
            for char in row_chars:
                ctk.CTkButton(
                    row_frame, text=char,
                    command=lambda ch=char: self._press(ch),
                    **_kw,
                ).pack(side='left', expand=True, fill='x', padx=2)

        # Ligne inférieure : Espace / ⌫ / ✓ OK
        bot = ctk.CTkFrame(self, fg_color='transparent')
        bot.pack(fill='x', padx=6, pady=(3, 6))

        ctk.CTkButton(
            bot, text='Espace',
            height=_KB_H, corner_radius=8,
            fg_color=_KB_BTN_BG, hover_color=_KB_HOV, text_color=TEXT,
            font=ctk.CTkFont(family='Helvetica', size=13),
            command=lambda: self._press(' '),
        ).pack(side='left', expand=True, fill='x', padx=2)

        ctk.CTkButton(
            bot, text='⌫',
            width=90, height=_KB_H, corner_radius=8,
            fg_color='#2A2035', hover_color='#3A1525', text_color='#FF6B6B',
            font=ctk.CTkFont(family='Helvetica', size=20),
            command=self._backspace,
        ).pack(side='left', padx=2)

        ctk.CTkButton(
            bot, text='✓  OK',
            width=110, height=_KB_H, corner_radius=8,
            fg_color=CYAN, hover_color='#00b8cc', text_color=BG,
            font=ctk.CTkFont(family='Helvetica', size=13, weight='bold'),
            command=self._validate,
        ).pack(side='left', padx=2)

    # ── Actions touches ───────────────────────────────────────────────────────
    def _press(self, char: str):
        if self._target is None:
            return
        try:
            inner = self._target._entry      # tk.Entry sous-jacent
            inner.insert('insert', char)
        except Exception:
            self._target.insert('end', char)

    def _backspace(self):
        if self._target is None:
            return
        try:
            inner = self._target._entry
            pos = inner.index('insert')
            if pos > 0:
                inner.delete(pos - 1, pos)
        except Exception:
            pass

    def _validate(self):
        if self._on_validate_cb:
            self._on_validate_cb()
        else:
            self.hide()

    # ── API publique ──────────────────────────────────────────────────────────
    def show(self, target: ctk.CTkEntry, on_validate=None):
        """Ancre le clavier en bas de la fenêtre parente et cible l'entrée."""
        self._target = target
        self._on_validate_cb = on_validate
        self.place(relx=0, rely=1.0, anchor='sw', relwidth=1.0)
        self.lift()

    def hide(self):
        self.place_forget()
        self._target = None
        self._on_validate_cb = None


class FaceRecognitionGUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title(config.GUI_WINDOW_TITLE)
        self.root.minsize(1100, 650)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        # Callbacks — liés depuis main.py
        self.on_enroll_user    = None
        self.on_motor_manual   = None
        self.on_calibrate_zero = None
        self.on_prev           = None
        self.on_toggle         = None
        self.on_next           = None
        self.on_scan_wifi      = None
        self.on_connect_wifi   = None
        self.on_reset_user       = None
        self.on_save_enroll      = None
        self.on_cancel_enroll    = None
        self.on_pair_bluetooth   = None

        self._active_page    = None
        self._pages          = {}
        self._last_img_size  = (0, 0)
        self.marker_frames: list = []
        self._current_marker     = None

        self._load_sidebar_icons()
        self._build_layout()
        self._vkb = CTkVirtualKeyboard(self.root)
        self._update_clock()
        logger.debug("GUI initialized.")

    # ── Icônes sidebar ────────────────────────────────────────────────────────
    def _load_sidebar_icons(self):
        """Charge les PNG en CTkImage 28×28. Fallback silencieux si absent."""
        self._sidebar_icons: dict = {}
        assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets'
        )
        for key, filename in [
            ('home',     'home.png'),
            ('vehicle',  'vehicle.png'),
            ('media',    'media.png'),
            ('settings', 'settings.png'),
        ]:
            path = os.path.join(assets_dir, filename)
            try:
                img = Image.open(path).resize((28, 28), Image.Resampling.LANCZOS)
                self._sidebar_icons[key] = ctk.CTkImage(
                    light_image=img, dark_image=img, size=(28, 28)
                )
                logger.debug(f"CTkImage chargée : {path}")
            except FileNotFoundError:
                logger.info(f"Icône absente, fallback emoji : {filename}")
            except Exception as e:
                logger.warning(f"Erreur chargement icône {filename} : {e}")

    # ── Layout racine ─────────────────────────────────────────────────────────
    def _build_layout(self):
        root_f = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        root_f.pack(fill='both', expand=True)
        root_f.columnconfigure(0, weight=0, minsize=130)
        root_f.columnconfigure(1, weight=1)
        root_f.rowconfigure(0, weight=1)

        self._build_sidebar(root_f)
        self._build_content_area(root_f)
        self._show_page('home')

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sb = ctk.CTkFrame(parent, fg_color=SIDEBAR, width=130, corner_radius=0)
        sb.grid(row=0, column=0, sticky='nsew')
        sb.pack_propagate(False)

        ctk.CTkLabel(sb, text="EV-OS", text_color=CYAN,
                     font=ctk.CTkFont(family='Helvetica', size=18, weight='bold')
                     ).pack(pady=(20, 2))
        ctk.CTkLabel(sb, text="Mirror AI", text_color=SUB,
                     font=ctk.CTkFont(family='Helvetica', size=9)
                     ).pack(pady=(0, 20))

        ctk.CTkFrame(sb, fg_color=CARD, height=1, corner_radius=0
                     ).pack(fill='x', padx=12, pady=(0, 16))

        self._nav_buttons = {}
        for key, emoji, label in [
            ('home',     '🏠', 'Home'),
            ('vehicle',  '🚗', 'Vehicle'),
            ('media',    '🎵', 'Media'),
            ('settings', '⚙️', 'Settings'),
        ]:
            icon_img = self._sidebar_icons.get(key)
            btn = ctk.CTkButton(
                sb,
                text=label,
                image=icon_img if icon_img else None,
                compound='top' if icon_img else 'left',
                width=110,
                height=70 if icon_img else 44,
                corner_radius=8,
                fg_color='transparent',
                text_color=SUB,
                hover_color=NAV_ACT,
                font=ctk.CTkFont(
                    family='Helvetica',
                    size=9 if icon_img else 11
                ),
                anchor='center',
                command=lambda k=key: self._show_page(k),
            )
            btn.pack(fill='x', padx=10, pady=2)
            self._nav_buttons[key] = btn

    def _show_page(self, name: str):
        if self._active_page == name:
            return
        if self._active_page:
            self._nav_buttons[self._active_page].configure(
                fg_color='transparent', text_color=SUB,
            )
        self._active_page = name
        self._nav_buttons[name].configure(
            fg_color=NAV_ACT, text_color=CYAN,
        )
        self._pages[name].tkraise()

    # ── Zone contenu ──────────────────────────────────────────────────────────
    def _build_content_area(self, parent):
        area = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0)
        area.grid(row=0, column=1, sticky='nsew')
        area.columnconfigure(0, weight=1)
        area.rowconfigure(0, weight=1)

        self._pages['home']     = self._make_home_page(area)
        self._pages['vehicle']  = self._make_vehicle_page(area)
        self._pages['media']    = self._make_media_page(area)
        self._pages['settings'] = self._make_settings_page(area)
        for p in self._pages.values():
            p.grid(row=0, column=0, sticky='nsew')

    # ── PAGE HOME ─────────────────────────────────────────────────────────────
    def _make_home_page(self, parent) -> ctk.CTkFrame:
        page = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(2, weight=1)

        # Row 0 — Topbar
        topbar = ctk.CTkFrame(page, fg_color=BG, corner_radius=0)
        topbar.grid(row=0, column=0, sticky='ew', padx=20, pady=(14, 2))
        topbar.columnconfigure(1, weight=1)

        left_info = ctk.CTkFrame(topbar, fg_color=BG, corner_radius=0)
        left_info.grid(row=0, column=0, sticky='w')
        self._clock_label = ctk.CTkLabel(
            left_info, text='',
            text_color=TEXT,
            font=ctk.CTkFont(family='Helvetica', size=22, weight='bold'),
        )
        self._clock_label.pack(anchor='w')
        self._date_label = ctk.CTkLabel(
            left_info, text='',
            text_color=SUB,
            font=ctk.CTkFont(family='Helvetica', size=10),
        )
        self._date_label.pack(anchor='w')

        right_info = ctk.CTkFrame(topbar, fg_color=BG, corner_radius=0)
        right_info.grid(row=0, column=2, sticky='e')

        # GIF animé position — tk.Label conservé pour le GC guard
        self._position_marker_label = tk.Label(right_info, bg=BG, bd=0)
        self._position_marker_label.pack(anchor='e', pady=(0, 2))

        self._weather_label = ctk.CTkLabel(
            right_info, text='📍 Belfort  19°C  ⛅',
            text_color=CYAN,
            font=ctk.CTkFont(family='Helvetica', size=11),
        )
        self._weather_label.pack(anchor='e')
        self._network_label = ctk.CTkLabel(
            right_info, text='📶 Connecté',
            text_color=CYAN,
            font=ctk.CTkFont(family='Helvetica', size=9),
        )
        self._network_label.pack(anchor='e')

        # Row 1 — Status strip
        self._status_label = ctk.CTkLabel(
            page, text='Initializing...', text_color=SUB, anchor='w',
            font=ctk.CTkFont(family='Helvetica', size=10),
        )
        self._status_label.grid(row=1, column=0, sticky='ew', padx=20, pady=(0, 4))

        # Row 2 — Carte GPS + overlays (map_container reste tk.Frame pour TkinterMapView)
        map_container = tk.Frame(page, bg=BLACK, bd=0)
        map_container.grid(row=2, column=0, sticky='nsew', padx=20, pady=(0, 6))
        self._map_container = map_container

        if _MAP_AVAILABLE:
            self._map_widget = TkinterMapView(map_container, corner_radius=0)
            self._map_widget.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._map_widget.set_tile_server(
                'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png')
            self._map_widget.set_position(47.642, 6.863)
            self._map_widget.set_zoom(12)
        else:
            tk.Label(map_container,
                     text='🗺  Carte GPS\n(pip install tkintermapview)',
                     bg='#0F1C28', fg=SUB, font=('Helvetica', 14),
                     ).place(relx=0.5, rely=0.5, anchor='center')
            self._map_widget = None

        # Overlay vidéo — tk.Frame + tk.Label pour l'affichage des frames caméra
        self._video_overlay = tk.Frame(map_container, bg=BLACK, bd=0)
        self._video_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.video_label = tk.Label(self._video_overlay, bg=BLACK)
        self.video_label.pack(fill='both', expand=True)

        # Barre de recherche CTk
        self._search_bar_frame = ctk.CTkFrame(
            map_container, fg_color=ENTRY_BG, corner_radius=20,
        )
        self._search_bar_frame.place(relx=0.5, y=12, anchor='n')

        self._search_entry = ctk.CTkEntry(
            self._search_bar_frame,
            width=280, height=36, corner_radius=20,
            fg_color='transparent', border_width=0,
            text_color=TEXT,
            placeholder_text='Rechercher une destination...',
            placeholder_text_color=SUB,
            font=ctk.CTkFont(family='Helvetica', size=11),
        )
        self._search_entry.pack(side='left', padx=(14, 0))
        self._search_entry.bind('<Return>', lambda e: (self._on_map_search(), self.hide_keyboard()))
        self._search_entry.bind('<FocusIn>', lambda e: self.show_keyboard(self._search_entry), add='+')

        ctk.CTkButton(
            self._search_bar_frame, text='⌕',
            width=36, height=36, corner_radius=18,
            fg_color='transparent', text_color=CYAN,
            hover_color=NAV_ACT,
            font=ctk.CTkFont(size=14),
            command=self._on_map_search,
        ).pack(side='left', padx=(2, 4))

        # Bouton "Changer de conducteur" — affiché après auth
        self._change_driver_btn = ctk.CTkButton(
            map_container, text='🔄  Changer de conducteur',
            width=220, height=38, corner_radius=19,
            fg_color=ENTRY_BG, text_color=SUB,
            hover_color=NAV_ACT,
            border_width=1, border_color='#2A3A4A',
            font=ctk.CTkFont(family='Helvetica', size=10),
            command=lambda: self.on_reset_user and self.on_reset_user(),
        )

        # Row 3 — Barre Spotify
        self._build_spotify_bar(page, row=3)

        return page

    def _build_spotify_bar(self, parent, row: int):
        bar = ctk.CTkFrame(parent, fg_color=CARD_CTK, corner_radius=12)
        bar.grid(row=row, column=0, sticky='ew', padx=20, pady=(0, 12))
        bar.columnconfigure(1, weight=1)

        # Contrôles transport
        ctrl = ctk.CTkFrame(bar, fg_color='transparent', corner_radius=0)
        ctrl.grid(row=0, column=0, padx=16, pady=14)

        _skip = dict(
            width=36, height=36, corner_radius=18,
            fg_color='transparent', hover_color=NAV_ACT,
            font=ctk.CTkFont(size=14),
        )
        ctk.CTkButton(ctrl, text='⏮', text_color=SUB,
                      command=lambda: self.on_prev and self.on_prev(),
                      **_skip).pack(side='left', padx=4)
        self._bar_toggle = ctk.CTkButton(
            ctrl, text='⏯', text_color=CYAN,
            width=40, height=40, corner_radius=20,
            fg_color=NAV_ACT, hover_color='#1E4A5A',
            font=ctk.CTkFont(size=16, weight='bold'),
            command=lambda: self.on_toggle and self.on_toggle(),
        )
        self._bar_toggle.pack(side='left', padx=6)
        ctk.CTkButton(ctrl, text='⏭', text_color=SUB,
                      command=lambda: self.on_next and self.on_next(),
                      **_skip).pack(side='left', padx=4)

        # Infos piste
        info = ctk.CTkFrame(bar, fg_color='transparent', corner_radius=0)
        info.grid(row=0, column=1, sticky='w', padx=8)
        self._bar_title = ctk.CTkLabel(
            info, text='En attente...', text_color=TEXT, anchor='w',
            font=ctk.CTkFont(family='Helvetica', size=11, weight='bold'),
        )
        self._bar_title.pack(anchor='w')
        self._bar_artist = ctk.CTkLabel(
            info, text='—', text_color=SUB, anchor='w',
            font=ctk.CTkFont(family='Helvetica', size=9),
        )
        self._bar_artist.pack(anchor='w')

        # Progression (décorative)
        prog_wrap = ctk.CTkFrame(bar, fg_color='transparent', corner_radius=0)
        prog_wrap.grid(row=0, column=2, padx=20, sticky='e')
        self._progress = ctk.CTkProgressBar(
            prog_wrap, width=90, height=4, corner_radius=2,
            fg_color='#1A2A3A', progress_color=CYAN,
        )
        self._progress.set(0.35)
        self._progress.pack(pady=16)

    # ── PAGE VEHICLE ──────────────────────────────────────────────────────────
    def _make_vehicle_page(self, parent) -> ctk.CTkFrame:
        page = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            page, text='Mirror Controls', text_color=TEXT,
            font=ctk.CTkFont(family='Helvetica', size=20, weight='bold'),
        ).grid(row=0, column=0, sticky='w', padx=24, pady=(20, 14))

        content = ctk.CTkFrame(page, fg_color=BG, corner_radius=0)
        content.grid(row=1, column=0, sticky='nsew', padx=24, pady=(0, 12))
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        # Carte sélecteur de rétroviseur
        sel = ctk.CTkFrame(content, corner_radius=15, fg_color=CARD_CTK)
        sel.grid(row=0, column=0, sticky='nsew', padx=(0, 14))

        ctk.CTkLabel(sel, text='Select Mirror', text_color=SUB,
                     font=ctk.CTkFont(family='Helvetica', size=10)
                     ).pack(pady=(20, 14))

        self._mirror_var = tk.StringVar(value='left')
        self._btn_left  = self._sel_btn(sel, '◀  Left',  'left')
        self._btn_right = self._sel_btn(sel, 'Right  ▶', 'right')
        self._refresh_mirror_btns()

        self._btn_right.configure(
            state='disabled',
            fg_color='#141E2B', text_color='#3A4A5A',
            border_color='#1E2A38', hover_color='#141E2B',
        )
        ctk.CTkLabel(sel, text='(Non connecté)', text_color='#3A4A5A',
                     fg_color='transparent',
                     font=ctk.CTkFont(family='Helvetica', size=9)
                     ).pack(pady=(0, 20))

        # Carte pavé directionnel
        dpad_outer = ctk.CTkFrame(content, corner_radius=15, fg_color=CARD_CTK)
        dpad_outer.grid(row=0, column=1, sticky='nsew')

        ctk.CTkLabel(dpad_outer, text='Mirror Alignment', text_color=SUB,
                     font=ctk.CTkFont(family='Helvetica', size=10)
                     ).pack(pady=(20, 14))

        self._make_dpad(dpad_outer, self._motor).pack()

        self._motor_counter_label = ctk.CTkLabel(
            dpad_outer, text='→ 0 ms   ↓ 0 ms',
            text_color=SUB,
            font=ctk.CTkFont(family='Helvetica', size=10),
        )
        self._motor_counter_label.pack(pady=(14, 20))

        return page

    def _sel_btn(self, parent, label: str, value: str) -> ctk.CTkButton:
        btn = ctk.CTkButton(
            parent, text=label,
            width=160, height=44, corner_radius=10,
            fg_color='transparent', text_color=SUB,
            border_width=1, border_color='#2A3A4A',
            hover_color=NAV_ACT,
            font=ctk.CTkFont(family='Helvetica', size=12),
            command=lambda v=value: self._select_mirror(v),
        )
        btn.pack(pady=5, padx=16)
        return btn

    def _select_mirror(self, value: str):
        self._mirror_var.set(value)
        self._refresh_mirror_btns()

    def _refresh_mirror_btns(self):
        active = self._mirror_var.get()
        if active == 'left':
            self._btn_left.configure(
                fg_color=CYAN, text_color=BG, border_color=CYAN,
                font=ctk.CTkFont(family='Helvetica', size=12, weight='bold'),
            )
        else:
            self._btn_left.configure(
                fg_color='transparent', text_color=SUB, border_color='#2A3A4A',
                font=ctk.CTkFont(family='Helvetica', size=12),
            )

    def _motor(self, direction: str):
        if self.on_motor_manual:
            self.on_motor_manual(direction)

    def _make_dpad(self, parent, cmd_fn) -> ctk.CTkFrame:
        """Pavé directionnel EV-OS — placement pixel par pixel via place().
        On abandonne grid() : CTkFrame possède un canvas interne qui court-circuite
        columnconfigure/uniform, ce qui rendait ◄ systématiquement plus étroit.
        """
        _BTN_BG = '#1E2D3D'
        _SZ   = 60           # côté du bouton carré
        _PAD  = 5            # espace entre boutons
        _CELL = _SZ + _PAD   # taille d'une cellule virtuelle (65 px)
        _W    = _CELL * 3    # largeur totale du frame (195 px)

        # Frame de taille fixe explicite — pack_propagate(False) empêche
        # les enfants de redimensionner le conteneur.
        frame = ctk.CTkFrame(parent, fg_color='transparent',
                             width=_W, height=_W)
        frame.pack_propagate(False)

        _BTN_KW = dict(
            width=_SZ, height=_SZ, corner_radius=10,
            fg_color=_BTN_BG, text_color=CYAN,
            hover_color=CYAN, border_width=0,
            anchor='center',
            font=ctk.CTkFont(family='Helvetica', size=22, weight='bold'),
        )

        # Centre de chaque cellule virtuelle
        def _cx(col: int) -> int: return col * _CELL + _CELL // 2
        def _cy(row: int) -> int: return row * _CELL + _CELL // 2

        btn_up    = ctk.CTkButton(frame, text='▲', command=lambda: cmd_fn('up'),    **_BTN_KW)
        btn_left  = ctk.CTkButton(frame, text='◀', command=lambda: cmd_fn('left'),  **_BTN_KW)
        btn_right = ctk.CTkButton(frame, text='►', command=lambda: cmd_fn('right'), **_BTN_KW)
        btn_down  = ctk.CTkButton(frame, text='▼', command=lambda: cmd_fn('down'),  **_BTN_KW)

        btn_up   .place(x=_cx(1), y=_cy(0), anchor='center')  # row 0 col 1
        btn_left .place(x=_cx(0), y=_cy(1), anchor='center')  # row 1 col 0
        btn_right.place(x=_cx(2), y=_cy(1), anchor='center')  # row 1 col 2
        btn_down .place(x=_cx(1), y=_cy(2), anchor='center')  # row 2 col 1

        ctk.CTkLabel(frame, text='◉', text_color='#2A3F52', fg_color=_BTN_BG,
                     width=_SZ, height=_SZ, anchor='center',
                     font=ctk.CTkFont(size=14),
                     ).place(x=_cx(1), y=_cy(1), anchor='center')  # row 1 col 1

        def _on_enter(e, b):
            try:
                b.configure(text_color=BG)
            except Exception:
                pass

        def _on_leave(e, b):
            try:
                b.configure(text_color=CYAN)
            except Exception:
                pass

        for btn in (btn_up, btn_left, btn_right, btn_down):
            btn.bind('<Enter>', lambda e, b=btn: _on_enter(e, b))
            btn.bind('<Leave>', lambda e, b=btn: _on_leave(e, b))

        return frame

    def show_enroll_panel(self, person_id: str):
        """Affiche le panneau d'assistant d'enrôlement dans Settings."""
        self._enroll_name_label.configure(text=f'Enrôlement : {person_id}')
        self._enroll_counter_label.configure(text='→ 0 ms   ↓ 0 ms')
        self._driver_register_btn.grid_remove()
        self._driver_separator.grid_remove()
        self._driver_zero_btn.grid_remove()
        self._enroll_panel.grid()

    def hide_enroll_panel(self):
        """Masque le panneau d'enrôlement et restaure les boutons principaux."""
        self._enroll_panel.grid_remove()
        self._driver_register_btn.grid()
        self._driver_separator.grid()
        self._driver_zero_btn.grid()

    def update_motor_counters(self, droite: int, bas: int):
        """Met à jour les compteurs (Vehicle + panneau enrôlement)."""
        text = f'→ {droite} ms   ↓ {bas} ms'
        if hasattr(self, '_motor_counter_label'):
            self._motor_counter_label.configure(text=text)
        if hasattr(self, '_enroll_counter_label'):
            self._enroll_counter_label.configure(text=text)

    # ── PAGE MEDIA ────────────────────────────────────────────────────────────
    def _make_media_page(self, parent) -> ctk.CTkFrame:
        page = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0)

        # ── Carte Bluetooth ───────────────────────────────────────────────────
        self._bt_card = ctk.CTkFrame(page, fg_color=CARD_CTK, corner_radius=12)
        self._bt_card.pack(fill='x', padx=20, pady=(18, 0))

        self._bt_status_label = ctk.CTkLabel(
            self._bt_card,
            text='📶  Bluetooth — en attente de connexion',
            text_color=SUB,
            wraplength=520,
            font=ctk.CTkFont(family='Helvetica', size=12),
        )
        self._bt_status_label.pack(side='left', padx=16, pady=12)

        self._bt_pair_btn = ctk.CTkButton(
            self._bt_card, text='🔗  Associer un téléphone',
            width=200, height=36, corner_radius=8,
            fg_color=CYAN, hover_color='#00b8cc', text_color=BG,
            font=ctk.CTkFont(family='Helvetica', size=11, weight='bold'),
            command=lambda: self.on_pair_bluetooth and self.on_pair_bluetooth(),
        )
        self._bt_pair_btn.pack(side='right', padx=12)

        # ── Zone lecteur (centrée dans l'espace restant) ──────────────────────
        self._bt_player_area = ctk.CTkFrame(page, fg_color=BG, corner_radius=0)
        self._bt_player_area.pack(fill='both', expand=True)
        player_area = self._bt_player_area   # alias local pour la suite

        # ── Toast de connexion (positionné sur page, initialement caché) ──────
        self._bt_flash = ctk.CTkFrame(page, fg_color='#0D2B1E', corner_radius=14)
        self._bt_flash_lbl = ctk.CTkLabel(
            self._bt_flash,
            text='',
            text_color='#00E676',
            font=ctk.CTkFont(family='Helvetica', size=15, weight='bold'),
        )
        self._bt_flash_lbl.pack(padx=24, pady=14)

        center = ctk.CTkFrame(player_area, fg_color=BG, corner_radius=0)
        center.place(relx=0.5, rely=0.5, anchor='center')

        # Album art
        self._media_art_frame = ctk.CTkFrame(
            center, fg_color=CARD_CTK, width=210, height=210, corner_radius=16,
        )
        self._media_art_frame.pack(pady=(0, 18))
        self._media_art_frame.pack_propagate(False)

        _ph_img = _make_placeholder_img(200)
        self._placeholder_ctk_img = ctk.CTkImage(
            light_image=_ph_img, dark_image=_ph_img, size=(200, 200)
        )
        self._media_art_label = ctk.CTkLabel(
            self._media_art_frame, text='', fg_color='transparent',
            image=self._placeholder_ctk_img,
        )
        self._media_art_label.place(relx=0.5, rely=0.5, anchor='center')
        self._last_art_url = None

        self._media_title = ctk.CTkLabel(
            center, text='En attente...', text_color=TEXT, wraplength=380,
            font=ctk.CTkFont(family='Helvetica', size=22, weight='bold'),
        )
        self._media_title.pack(pady=(0, 4))
        self._media_artist = ctk.CTkLabel(
            center, text='—', text_color=SUB,
            font=ctk.CTkFont(family='Helvetica', size=13),
        )
        self._media_artist.pack(pady=(0, 28))

        # Contrôles
        ctrl = ctk.CTkFrame(center, fg_color='transparent', corner_radius=0)
        ctrl.pack()

        _skip = dict(
            width=56, height=56, corner_radius=28,
            fg_color='transparent', hover_color=NAV_ACT,
            font=ctk.CTkFont(size=22),
        )
        ctk.CTkButton(ctrl, text='⏮', text_color=SUB,
                      command=lambda: self.on_prev and self.on_prev(),
                      **_skip).pack(side='left', padx=14)
        self._media_toggle = ctk.CTkButton(
            ctrl, text='▶', text_color=BG,
            width=68, height=68, corner_radius=34,
            fg_color=CYAN, hover_color='#00b8cc',
            font=ctk.CTkFont(size=28, weight='bold'),
            command=lambda: self.on_toggle and self.on_toggle(),
        )
        self._media_toggle.pack(side='left', padx=14)
        ctk.CTkButton(ctrl, text='⏭', text_color=SUB,
                      command=lambda: self.on_next and self.on_next(),
                      **_skip).pack(side='left', padx=14)

        return page

    # ── PAGE SETTINGS ─────────────────────────────────────────────────────────
    def _make_settings_page(self, parent) -> ctk.CTkFrame:
        page = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0)
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(1, weight=1)

        ctk.CTkLabel(page, text='  Settings', text_color=TEXT,
                     font=ctk.CTkFont(family='Helvetica', size=18, weight='bold')
                     ).grid(row=0, column=0, columnspan=2, sticky='w',
                            padx=24, pady=(20, 14))

        # ── Carte Wi-Fi ────────────────────────────────────────────────────────
        wifi_card = ctk.CTkFrame(page, corner_radius=15, fg_color=CARD_CTK)
        wifi_card.grid(row=1, column=0, sticky='nsew', padx=(24, 8), pady=(0, 20))
        wifi_card.columnconfigure(0, weight=1)
        wifi_card.rowconfigure(2, weight=1)

        ctk.CTkLabel(wifi_card, text='🌐  Configuration Wi-Fi',
                     text_color=CYAN,
                     font=ctk.CTkFont(family='Helvetica', size=12, weight='bold')
                     ).grid(row=0, column=0, sticky='w', padx=16, pady=(14, 8))

        ctk.CTkButton(wifi_card, text='🔍  Scanner les réseaux disponibles',
                      height=42, corner_radius=10,
                      fg_color=NAV_ACT, text_color=CYAN,
                      hover_color='#1E4A5A',
                      font=ctk.CTkFont(family='Helvetica', size=11, weight='bold'),
                      command=lambda: self.on_scan_wifi and self.on_scan_wifi()
                      ).grid(row=1, column=0, sticky='ew', padx=16, pady=(0, 8))

        # Listbox Wi-Fi (tk.Listbox — aucun équivalent CTk disponible)
        list_frame = tk.Frame(wifi_card, bg=ENTRY_BG, bd=0)
        list_frame.grid(row=2, column=0, sticky='nsew', padx=16, pady=(0, 8))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._wifi_listbox = tk.Listbox(
            list_frame,
            bg=ENTRY_BG, fg=TEXT, selectbackground=NAV_ACT, selectforeground=CYAN,
            font=('Helvetica', 11), bd=0, highlightthickness=1,
            highlightcolor=CYAN, highlightbackground=ENTRY_BG,
            activestyle='none',
        )
        self._wifi_listbox.grid(row=0, column=0, sticky='nsew')

        scroll = ttk.Scrollbar(list_frame, orient='vertical',
                               command=self._wifi_listbox.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        self._wifi_listbox.config(yscrollcommand=scroll.set)

        # Mot de passe (CTkEntry)
        ctk.CTkLabel(wifi_card, text='Mot de passe :',
                     text_color=SUB,
                     font=ctk.CTkFont(family='Helvetica', size=10)
                     ).grid(row=3, column=0, sticky='w', padx=16, pady=(4, 0))

        self._wifi_pw_entry = ctk.CTkEntry(
            wifi_card,
            show='*', height=38, corner_radius=10,
            fg_color=ENTRY_BG, border_color='#2A3A4A',
            text_color=TEXT,
            placeholder_text='••••••••',
            font=ctk.CTkFont(family='Helvetica', size=11),
        )
        self._wifi_pw_entry.grid(row=4, column=0, sticky='ew',
                                 padx=16, pady=(4, 8))
        self._wifi_pw_entry.bind('<FocusIn>', lambda e: self.show_keyboard(self._wifi_pw_entry), add='+')
        self._wifi_pw_entry.bind('<Return>',  lambda e: self.hide_keyboard(), add='+')

        ctk.CTkButton(wifi_card, text='⚡  Connexion',
                      height=44, corner_radius=10,
                      fg_color=CYAN, text_color=BG,
                      hover_color='#00b8cc',
                      font=ctk.CTkFont(family='Helvetica', size=12, weight='bold'),
                      command=lambda: self.on_connect_wifi and self.on_connect_wifi()
                      ).grid(row=5, column=0, sticky='ew', padx=16, pady=(0, 8))

        self._wifi_status_label = ctk.CTkLabel(
            wifi_card, text='', text_color=SUB, anchor='w', wraplength=300,
            font=ctk.CTkFont(family='Helvetica', size=10),
        )
        self._wifi_status_label.grid(row=6, column=0, sticky='ew',
                                     padx=16, pady=(0, 12))

        # ── Carte Conducteurs ──────────────────────────────────────────────────
        driver_card = ctk.CTkFrame(page, corner_radius=15, fg_color=CARD_CTK)
        driver_card.grid(row=1, column=1, sticky='nsew', padx=(8, 24), pady=(0, 20))
        driver_card.columnconfigure(1, weight=1)

        ctk.CTkLabel(driver_card, text='👤  Gestion des Conducteurs',
                     text_color=CYAN,
                     font=ctk.CTkFont(family='Helvetica', size=12, weight='bold')
                     ).grid(row=0, column=0, columnspan=2, sticky='w',
                            padx=16, pady=(14, 16))

        self._driver_register_btn = ctk.CTkButton(
            driver_card, text='✅  Enregistrer Nouveau Conducteur',
            height=46, corner_radius=10,
            fg_color=CYAN, text_color=BG, hover_color='#00b8cc',
            font=ctk.CTkFont(family='Helvetica', size=11, weight='bold'),
            command=lambda: self.on_enroll_user and self.on_enroll_user()
        )
        self._driver_register_btn.grid(row=1, column=0, columnspan=2,
                                       sticky='ew', padx=16, pady=(0, 12))

        self._driver_separator = ctk.CTkFrame(
            driver_card, fg_color=NAV_ACT, height=1, corner_radius=0,
        )
        self._driver_separator.grid(row=2, column=0, columnspan=2,
                                    sticky='ew', padx=16, pady=(4, 12))

        self._driver_zero_btn = ctk.CTkButton(
            driver_card, text='🎯  Fixer le Point Zéro Moteurs',
            height=40, corner_radius=10,
            fg_color='transparent', text_color=SUB,
            border_width=1, border_color='#2A3A4A', hover_color=NAV_ACT,
            font=ctk.CTkFont(family='Helvetica', size=10),
            command=lambda: self.on_calibrate_zero and self.on_calibrate_zero()
        )
        self._driver_zero_btn.grid(row=3, column=0, columnspan=2,
                                   sticky='ew', padx=16, pady=(0, 16))

        # ── Panneau assistant d'enrôlement (caché par défaut) ─────────────────
        self._enroll_panel = ctk.CTkFrame(driver_card, fg_color=CARD_CTK,
                                          corner_radius=0)
        self._enroll_panel.grid(row=4, column=0, columnspan=2, sticky='ew')
        self._enroll_panel.grid_remove()

        self._enroll_name_label = ctk.CTkLabel(
            self._enroll_panel, text='Enrôlement : —',
            text_color=CYAN,
            font=ctk.CTkFont(family='Helvetica', size=11, weight='bold'),
        )
        self._enroll_name_label.pack(pady=(12, 10))

        self._make_dpad(self._enroll_panel, self._motor).pack()

        self._enroll_counter_label = ctk.CTkLabel(
            self._enroll_panel, text='→ 0 ms   ↓ 0 ms',
            text_color=SUB,
            font=ctk.CTkFont(family='Helvetica', size=10),
        )
        self._enroll_counter_label.pack(pady=(8, 10))

        ctk.CTkButton(self._enroll_panel, text='💾  Sauvegarder le Profil',
                      height=44, corner_radius=10,
                      fg_color=CYAN, text_color=BG, hover_color='#00b8cc',
                      font=ctk.CTkFont(family='Helvetica', size=11, weight='bold'),
                      command=lambda: self.on_save_enroll and self.on_save_enroll()
                      ).pack(fill='x', padx=16, pady=(0, 8))

        ctk.CTkButton(self._enroll_panel, text='✖  Annuler',
                      height=38, corner_radius=10,
                      fg_color='transparent', text_color=SUB,
                      border_width=1, border_color='#2A3A4A', hover_color=NAV_ACT,
                      font=ctk.CTkFont(family='Helvetica', size=10),
                      command=lambda: self.on_cancel_enroll and self.on_cancel_enroll()
                      ).pack(fill='x', padx=16, pady=(0, 12))

        return page

    # ── Recherche carte ───────────────────────────────────────────────────────
    def _on_search_focus_in(self, _event):
        pass  # CTkEntry gère le placeholder nativement

    def _on_search_focus_out(self, _event):
        pass  # CTkEntry gère le placeholder nativement

    def _on_map_search(self):
        if not self._map_widget or not _geocoder:
            return
        query = self._search_entry.get().strip()
        if not query:
            return
        try:
            result = _geocoder.osm(query, headers={'User-Agent': 'SbarroCockpitApp/1.0'})
            if result.ok:
                self._map_widget.set_position(result.lat, result.lng)
                self._map_widget.set_zoom(13)
            else:
                logger.warning(f"_on_map_search : aucun résultat pour « {query} »")
        except Exception as e:
            logger.warning(f"_on_map_search : {e}")

    # ── Horloge ───────────────────────────────────────────────────────────────
    def _update_clock(self):
        now = datetime.now()
        self._clock_label.configure(text=now.strftime('%H:%M:%S'))
        self._date_label.configure(text=now.strftime('%A %d %B %Y'))
        self.root.after(1000, self._update_clock)

    # ── Public API ────────────────────────────────────────────────────────────
    def update_weather(self, text: str):
        self._weather_label.configure(text=text)

    def update_network_status(self, online: bool):
        if online:
            self._network_label.configure(text='📶 Connecté', text_color=CYAN)
        else:
            self._network_label.configure(
                text='⚠️ Hors ligne — Activez le partage 4G/5G',
                text_color='#FF4444',
            )

    def update_wifi_list(self, networks: list):
        self._wifi_listbox.delete(0, 'end')
        for net in networks:
            self._wifi_listbox.insert('end', f'  {net}')

    def get_selected_ssid(self) -> str:
        sel = self._wifi_listbox.curselection()
        return self._wifi_listbox.get(sel[0]).strip() if sel else ''

    def get_wifi_password(self) -> str:
        return self._wifi_pw_entry.get()

    def update_wifi_status(self, text: str, error: bool = False):
        self._wifi_status_label.configure(
            text=text, text_color='#FF4444' if error else CYAN,
        )

    def update_bt_status(self, text: str, active: bool = False):
        """Met à jour le label de statut Bluetooth.
        active=True : texte en cyan (mode appairage) ; False : gris (veille).
        """
        if hasattr(self, '_bt_status_label'):
            self._bt_status_label.configure(
                text=text,
                text_color=CYAN if active else SUB,
            )

    # ── Pochette d'album ──────────────────────────────────────────────────────

    def update_cover_art(self, url: str):
        """Déclenche le chargement de la pochette en arrière-plan.
        Appel sans-effet si l'URL n'a pas changé depuis le dernier cycle.
        """
        if url == self._last_art_url:
            return
        self._last_art_url = url
        import threading
        threading.Thread(target=self._load_cover_art, args=(url,), daemon=True).start()

    def _load_cover_art(self, url: str):
        """Télécharge / ouvre l'image et met à jour le label via root.after().
        Jamais appelé depuis le thread Tkinter.
        """
        try:
            if not url:
                raise ValueError("no url")
            if url.startswith('file://'):
                path = url[len('file://'):]
                img  = Image.open(path)
            elif url.startswith(('http://', 'https://')):
                with urllib.request.urlopen(url, timeout=4) as resp:
                    img = Image.open(_io.BytesIO(resp.read()))
            else:
                raise ValueError(f"format non supporté : {url}")
            img      = img.convert('RGB').resize((200, 200), Image.LANCZOS)
            ctk_img  = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
        except Exception:
            ctk_img = self._placeholder_ctk_img
        self.root.after(0, lambda i=ctk_img: self._set_cover_art(i))

    def _set_cover_art(self, ctk_img: ctk.CTkImage):
        """Applique l'image au label (toujours sur le thread Tkinter)."""
        if hasattr(self, '_media_art_label'):
            self._media_art_label.configure(image=ctk_img)

    def set_bt_connected(self, first_connection: bool = False):
        """Cache la carte BT et affiche le toast si première connexion."""
        if hasattr(self, '_bt_card'):
            self._bt_card.pack_forget()
        if first_connection:
            self._show_bt_flash("✅  Téléphone connecté !")

    def set_bt_disconnected(self):
        """Réaffiche la carte BT avec le label initial."""
        if hasattr(self, '_bt_card') and hasattr(self, '_bt_player_area'):
            self._bt_status_label.configure(
                text='📶  Bluetooth — en attente de connexion',
                text_color=SUB,
            )
            self._bt_card.pack(
                fill='x', padx=20, pady=(18, 0),
                before=self._bt_player_area,
            )

    def _show_bt_flash(self, text: str):
        """Affiche un toast vert centré sur la page Média pendant 3 secondes."""
        if not hasattr(self, '_bt_flash'):
            return
        self._bt_flash_lbl.configure(text=text)
        self._bt_flash.place(relx=0.5, rely=0.08, anchor='n')
        self._bt_flash.lift()
        self._bt_flash.after(3000, self._bt_flash.place_forget)

    def update_status(self, text: str):
        if not isinstance(text, str):
            text = str(text)
        self._status_label.configure(text=text)

    def update_track_ui(self, title: str, artist: str, is_playing: bool):
        icon = '⏸' if is_playing else '▶'
        self._bar_title.configure(text=title)
        self._bar_artist.configure(text=artist)
        self._bar_toggle.configure(text=icon)
        self._media_title.configure(text=title)
        self._media_artist.configure(text=artist)
        self._media_toggle.configure(text=icon)

    def update_image(self, frame: Optional[np.ndarray]):
        if frame is None:
            self.video_label.config(image='')
            self.video_label.image = None
            return
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            logger.warning('Invalid frame passed to update_image.')
            return
        try:
            w = self.video_label.winfo_width()
            h = self.video_label.winfo_height()
            if w <= 10 or h <= 10:
                w, h = 640, 480
            ih, iw = frame.shape[:2]
            scale  = min(w / iw, h / ih)
            nw     = max(1, int(iw * scale))
            nh     = max(1, int(ih * scale))
            if abs(nw - self._last_img_size[0]) > 2 or abs(nh - self._last_img_size[1]) > 2:
                self._last_img_size = (nw, nh)
            resized = cv2.resize(frame, self._last_img_size, interpolation=cv2.INTER_AREA)
            imgtk   = ImageTk.PhotoImage(
                image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk  # GC guard
        except Exception as e:
            logger.error(f'update_image error: {e}', exc_info=True)
            self.video_label.config(image='')
            self.video_label.image = None

    # ── GIF marqueur de position ──────────────────────────────────────────────
    def load_animated_gif(self, path: str):
        """Extrait toutes les frames du GIF redimensionnées en 40×40."""
        self.marker_frames = []
        try:
            gif = Image.open(path)
            while True:
                frame = gif.copy().convert('RGBA').resize(
                    (40, 40), Image.Resampling.LANCZOS)
                self.marker_frames.append(ImageTk.PhotoImage(frame))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        except FileNotFoundError:
            logger.info(f'GIF marqueur absent : {path}')
        except Exception as e:
            logger.warning(f'load_animated_gif : {e}')
        if self.marker_frames:
            logger.info(f'GIF chargé : {len(self.marker_frames)} frame(s) — {path}')

    def animate_marker(self, frame_index: int = 0):
        """Boucle GIF 100 ms sur le label topbar."""
        if not self.marker_frames:
            return
        frame = self.marker_frames[frame_index]
        self._position_marker_label.config(image=frame)
        self._position_marker_label.image = frame  # GC guard
        self.root.after(100, self.animate_marker,
                        (frame_index + 1) % len(self.marker_frames))

    def show_map_fullscreen(self):
        """Cache l'overlay vidéo — carte GPS en plein écran."""
        self._video_overlay.place_forget()
        self._change_driver_btn.place(relx=1.0, rely=1.0, anchor='se', x=-14, y=-14)

    def show_video_feed(self):
        """Restaure l'overlay vidéo sur la carte."""
        self.video_label.config(image='')
        self.video_label.image = None
        self._change_driver_btn.place_forget()
        self._video_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

    # ── Clavier virtuel ───────────────────────────────────────────────────────
    def show_keyboard(self, target: ctk.CTkEntry, on_validate=None):
        """Affiche le clavier virtuel ciblant *target*."""
        self._vkb.show(target, on_validate=on_validate)

    def hide_keyboard(self):
        """Masque le clavier virtuel."""
        self._vkb.hide()

    def prompt_driver_name(self) -> Optional[str]:
        """Panneau de saisie conducteur in-app avec clavier virtuel intégré.
        Bloque le thread Tkinter via wait_variable() (événements toujours traités).
        """
        result_var = tk.StringVar(value='')
        done_var   = tk.BooleanVar(value=False)

        # Fond plein écran (sans transparence — CTkFrame est opaque)
        overlay = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Carte de saisie centrée dans la moitié haute (laisse place au clavier)
        card = ctk.CTkFrame(overlay, fg_color=CARD_CTK, corner_radius=16)
        card.place(relx=0.5, rely=0.30, anchor='center')

        ctk.CTkLabel(
            card,
            text='Nouveau Conducteur',
            text_color=CYAN,
            font=ctk.CTkFont(family='Helvetica', size=18, weight='bold'),
        ).pack(padx=36, pady=(22, 10))

        entry = ctk.CTkEntry(
            card, width=360, height=46, corner_radius=10,
            fg_color=ENTRY_BG, border_color=CYAN, border_width=2,
            text_color=TEXT,
            placeholder_text='Nom du conducteur...',
            font=ctk.CTkFont(family='Helvetica', size=14),
        )
        entry.pack(padx=36, pady=(0, 16))

        btn_row = ctk.CTkFrame(card, fg_color='transparent')
        btn_row.pack(padx=36, pady=(0, 22), fill='x')

        def _confirm(*_):
            name = entry.get().strip()
            result_var.set(name)
            done_var.set(True)
            overlay.destroy()
            self.hide_keyboard()

        def _cancel(*_):
            done_var.set(True)
            overlay.destroy()
            self.hide_keyboard()

        entry.bind('<Return>', _confirm)

        ctk.CTkButton(
            btn_row, text='✗  Annuler',
            width=130, height=40, corner_radius=10,
            fg_color='#1E2D3D', hover_color='#3A2A2A', text_color=SUB,
            font=ctk.CTkFont(family='Helvetica', size=12),
            command=_cancel,
        ).pack(side='left', padx=(0, 10))

        ctk.CTkButton(
            btn_row, text='✓  Valider',
            height=40, corner_radius=10,
            fg_color=CYAN, hover_color='#00b8cc', text_color=BG,
            font=ctk.CTkFont(family='Helvetica', size=12, weight='bold'),
            command=_confirm,
        ).pack(side='left', expand=True, fill='x')

        # Affiche le clavier ciblant ce champ ; ✓ OK du clavier confirme aussi
        self.show_keyboard(entry, on_validate=_confirm)
        self._vkb.lift()   # clavier au-dessus de l'overlay opaque

        entry.focus_set()
        self.root.wait_variable(done_var)
        return result_var.get() or None

    def set_map_position(self, lat: float, lon: float, label: str = ''):
        """Centre la carte et pose un marqueur."""
        if not self._map_widget:
            return
        try:
            self._map_widget.delete_all_marker()
            self._current_marker = None
            self._map_widget.set_position(lat, lon)
            self._map_widget.set_zoom(14)
            if label:
                self._current_marker = self._map_widget.set_marker(
                    lat, lon, text=label)
        except Exception as e:
            logger.warning(f'set_map_position : {e}')
