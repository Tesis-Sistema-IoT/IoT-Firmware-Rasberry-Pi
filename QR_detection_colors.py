#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QR_detection
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pyzbar.pyzbar import ZBarSymbol, decode

Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


# ============================================================================
# ARGUMENTOS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QR de producción para Raspberry Pi")

    # Configuración básica de fuente, resolución y frecuencia de escaneo.
    parser.add_argument("--source", required=True, help="picamera0 | usb0 | video.mp4")
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--decode_width", type=int, default=480)
    parser.add_argument("--scan_every", type=int, default=5)
    parser.add_argument("--max_lost", type=int, default=150)
    parser.add_argument("--print_every", type=float, default=1.0)

    # Configuración visual: cámara en ventana o modo solo consola/logs.
    parser.add_argument("--view", default="camera", choices=["camera", "logs"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--log_only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs de consola para no frenar el loop.")
    parser.add_argument("--no_console_logs", action="store_true", help="Alias de --quiet.")

    # Configuración del LED RGB de estado.
    parser.add_argument("--led_disable", action="store_true", help="Desactivar el LED RGB de estado.")
    parser.add_argument("--led_pins", default="17,27,22", help="GPIO BCM para R,G,B")
    parser.add_argument("--led_common_anode", action="store_true", help="Usar si el LED RGB es ánodo común.")
    parser.add_argument("--led_color_startup", default="#000060", help="Color al encender/inicializar")
    parser.add_argument("--led_color_login_ok", default="#0000FF", help="Color cuando login API OK")
    parser.add_argument("--led_color_auth_ok", default="#00FF00", help="Color cuando el sistema ya terminó de cargar y está funcional.")
    parser.add_argument("--led_color_qr_1", default="#00FFFF", help="Color cuando ve 1 código QR")
    parser.add_argument("--led_color_qr_2", default="#8000FF", help="Color cuando ve 2 códigos QR")
    parser.add_argument("--led_color_qr_3", default="#FFFFFF", help="Color cuando ve 3 códigos QR")
    parser.add_argument("--led_color_qr_4", default="#FF00FF", help="Color cuando ve 4 códigos QR")
    parser.add_argument("--led_color_qr_5_plus", default="#FF0000", help="Color cuando ve 5 o más códigos QR")
    parser.add_argument("--led_color_error", default="#FFFFFF", help="Color para errores de LED o cierre")

    # Configuración del dibujo/tracking visual.
    parser.add_argument("--draw_trail", action="store_true")
    parser.add_argument("--trail_seconds", type=float, default=30.0)
    parser.add_argument("--trail_len", type=int, default=60)
    parser.add_argument("--track_match_dist", type=float, default=200.0)
    parser.add_argument("--draw_yolo_candidates", action="store_true")

    # Configuración de detección: pyzbar/OpenCV rápido y YOLO como fallback.
    parser.add_argument("--detector", default="yolo", choices=["pyzbar", "yolo"])
    parser.add_argument("--primary_detector", default="hybrid", choices=["pyzbar", "opencv", "hybrid"],
                        help="Detector rápido inicial. hybrid = pyzbar para leer + OpenCV para detectar presencia.")
    parser.add_argument("--model", default="")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max_det", type=int, default=20)
    parser.add_argument("--yolo_decode_crop", action="store_true")
    parser.add_argument("--yolo_pad", type=int, default=45)
    parser.add_argument("--yolo_imgsz", type=int, default=480)
    parser.add_argument("--yolo_no_pyzbar_first", action="store_true")
    parser.add_argument("--fast_yolo", action="store_true", help="Salta detector rápido y usa YOLO directo.")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--yolo_fallback_mode", default="presence", choices=["presence", "always", "never"],
                        help="presence recomendado: YOLO solo si OpenCV ve QR no leído. always: YOLO si pyzbar falla.")
    parser.add_argument("--yolo_decode_topk", type=int, default=3,
                        help="Máximo de candidatos YOLO a decodificar con pyzbar. Evita crop_ms gigante.")
    parser.add_argument("--yolo_min_size", type=int, default=18,
                        help="Descarta cajas YOLO demasiado pequeñas.")
    parser.add_argument("--yolo_min_area_ratio", type=float, default=0.0004,
                        help="Área mínima de caja YOLO respecto al frame.")
    parser.add_argument("--yolo_max_area_ratio", type=float, default=0.35,
                        help="Área máxima de caja YOLO respecto al frame.")
    parser.add_argument("--yolo_max_aspect", type=float, default=2.5,
                        help="Descarta cajas muy alargadas; QR debería ser casi cuadrado.")

    # Configuración para mejorar lectura en imágenes oscuras o difíciles.
    parser.add_argument("--low_light_enhance", action="store_true")
    parser.add_argument("--strong_crop_decode", action="store_true",
                        help="Solo aplica a crops, no al frame completo.")
    parser.add_argument("--full_strong_decode", action="store_true",
                        help="Lento: aplica variantes fuertes al frame completo. Normalmente NO usar.")

    # Configuración de métricas impresas o dibujadas en pantalla.
    parser.add_argument("--metrics_overlay", action="store_true")
    parser.add_argument("--metrics_window", type=int, default=60)
    parser.add_argument("--metrics_log_every", type=float, default=1.0)

    # Configuración de API/backend: login, URL de envío, timeouts y payload.
    parser.add_argument("--backend_url", default="")
    parser.add_argument("--backend_timeout", type=float, default=2.0)
    parser.add_argument("--auth_url", default="https://tesis-backend-service.onrender.com/iot/v1/auth/login/camera")
    parser.add_argument("--auth_payload", default="")
    parser.add_argument("--auth_payload_env", default="CAMERA_AUTH_PAYLOAD")
    parser.add_argument("--auth_timeout", type=float, default=50.0)

    # Configuración específica de cámara, formato de color, FPS y exposición.
    parser.add_argument("--color_fix", default="none", choices=["none", "rgb2bgr", "bgr2rgb"])
    parser.add_argument("--picamera_format", default="RGB888", choices=["RGB888", "BGR888", "XRGB888"])
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--exposure_time", type=int, default=0)
    parser.add_argument("--analogue_gain", type=float, default=0.0)

    return parser.parse_args()


# ============================================================================
# UTILIDADES
# ============================================================================

# Obtener los ms actuales
def now_ms() -> float:
    return time.perf_counter() * 1000.0

# Promedio seguro de una lista de valores
def avg(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0

# Percentil 95 de una lista de valores
def p95(values) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), 95))


# Imprime solo si no estamos en modo quiet/log_only, para evitar frenar el loop con prints.
def cprint(args: argparse.Namespace, *items, force: bool = False) -> None:
    if not getattr(args, "quiet", False):
        print(*items)


# Estos son los únicos logs que sí queremos ver aunque esté activo --quiet.
def prod_log(*items) -> None:
    print(*items, flush=True)


# Determina si las métricas deben ser calculadas y mostradas, basado en los argumentos de configuración.
def metrics_enabled(args: argparse.Namespace) -> bool:
    return not (getattr(args, "quiet", False) and getattr(args, "view", "camera") == "logs")


# Nombre del modelo sin ruta, para mostrar en logs métricas.
def safe_model_name(path: str) -> str:
    return Path(path).name if path else "none"

# Convierte resolución WIDTHxHEIGHT a tupla (WIDTH, HEIGHT)
def parse_resolution(resolution: str) -> Tuple[int, int]:
    try:
        w, h = map(int, resolution.lower().split("x"))
        return w, h
    except Exception as exc:
        raise RuntimeError("[INIT ERROR] --resolution debe ser WIDTHxHEIGHT, por ejemplo 1280x720") from exc


# ============================================================================
# MÉTRICAS
# ============================================================================

# Estructura para almacenar métricas detalladas de cada escaneo
@dataclass
class ScanMetrics:
    path: str = "-"
    total_ms: float = 0.0
    pyzbar_ms: float = 0.0
    opencv_ms: float = 0.0
    preprocess_ms: float = 0.0
    yolo_wall_ms: float = 0.0
    yolo_reported_infer_ms: float = 0.0
    crop_decode_ms: float = 0.0
    crop_count: int = 0
    candidate_count: int = 0
    decoded_count: int = 0
    rejected_count: int = 0
    confidences: List[float] = field(default_factory=list)

# Clase para almacenar métricas globales y promedios móviles,
class Metrics:
    def __init__(self, window: int = 60) -> None:
        self.started = time.perf_counter()
        self.frames = 0
        self.scans = 0
        self.candidates = 0
        self.decoded = 0
        self.rejected = 0
        self.last_scan = ScanMetrics()

        # Se quedan como listas vacías por compatibilidad con el resumen final y funciones visuales.
        self.capture_ms: List[float] = []
        self.loop_ms: List[float] = []
        self.scan_ms: List[float] = []
        self.pyzbar_ms: List[float] = []
        self.opencv_ms: List[float] = []
        self.yolo_ms: List[float] = []
        self.crop_ms: List[float] = []
        self.backend_ms: List[float] = []
        self.conf: List[float] = []
        self.instant_fps: List[float] = []

    # En producción no guardamos tiempo de captura por frame.
    def record_capture(self, ms: float) -> None:
        return

    # Solo contamos frames para no estar calculando FPS todo el tiempo.
    def record_frame(self, ms: float = 0.0) -> None:
        self.frames += 1

    # Solo guardamos contadores básicos del último escaneo.
    def record_scan(self, s: ScanMetrics) -> None:
        self.last_scan = s
        self.scans += 1
        self.candidates += s.candidate_count
        self.decoded += s.decoded_count
        self.rejected += s.rejected_count

    # En producción no se guardan métricas del backend, solo se imprime DB OK o DB ERROR.
    def record_backend(self, ms: float) -> None:
        return

    def fps(self) -> float:
        return 0.0

    def runtime_s(self) -> float:
        return max(0.001, time.perf_counter() - self.started)

    def scan_fps(self) -> float:
        return self.scans / self.runtime_s()

    # --quiet No se genera línea de métricas para journalctl. 
    def log_line(self, args: argparse.Namespace, tracker: "QRTracker") -> str:
        return "[METRICS DISABLED]"

    # --quiet No se generan métricas detalladas para journalctl. 
    def snapshot(self, args: argparse.Namespace, tracker: "QRTracker") -> dict:
        return {}


# ============================================================================
# LOGIN Y BACKEND
# ============================================================================

# Resuelve el payload de autenticación, dando prioridad a la variable de entorno si está configurada, o al argumento directo si no lo esta
def resolved_auth_payload(args: argparse.Namespace) -> str:
    from_env = os.getenv(args.auth_payload_env, "").strip() if args.auth_payload_env else ""
    return from_env or args.auth_payload.strip()


# Realiza el login a la cámara usando la URL y payload configurados
def login_camera(args: argparse.Namespace) -> Optional[str]:
    if not args.auth_url:
        cprint(args, "[AUTH] auth_url vacío; login omitido.")
        return None
    payload_raw = resolved_auth_payload(args)
    if not payload_raw:
        raise RuntimeError("[AUTH ERROR] Falta auth payload. Usa --auth_payload o CAMERA_AUTH_PAYLOAD.")
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("[AUTH ERROR] auth payload no es JSON válido.") from exc

    req = urllib.request.Request(
        args.auth_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = now_ms()
    try:
        with urllib.request.urlopen(req, timeout=args.auth_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            if not data.get("success"):
                raise RuntimeError("[AUTH ERROR] Login no exitoso.")
            token = data.get("data", {}).get("token")
            if not token:
                raise RuntimeError("[AUTH ERROR] Respuesta sin data.token.")
            prod_log(f"[AUTH OK] login_ms={now_ms() - started:.1f}")
            return token
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"[AUTH HTTP ERROR] status={error.code} body={body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"[AUTH URL ERROR] {error}") from error

# Clase que maneja el envío de datos al backend/API en un hilo separado, con una cola para evitar bloquear el loop principal, y registra métricas de tiempo de respuesta.
class BackendWorker:
    # inicializa el objeto con la configuraciónes basicas
    def __init__(self, args: argparse.Namespace, token: Optional[str], metrics: Metrics) -> None:
        self.args = args
        self.url = args.backend_url
        self.token = token
        self.timeout = args.backend_timeout
        self.metrics = metrics
        self.q: "queue.Queue[Tuple[int, str]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    # Arranca el hilo del worker si la URL de backend está configurada
    def start(self) -> None:
        if self.url:
            cprint(self.args, "[BACKEND] worker_start", force=True)
            self.thread.start()

    # Detiene el hilo del worker de forma segura
    def stop(self) -> None:
        if self.url:
            self.stop_event.set()
            self.q.put((-1, "__STOP__"))
            self.thread.join(timeout=2.0)
            cprint(self.args, "[BACKEND] worker_stop", force=True)

    # Agrega un nuevo dato a la cola para enviar al backend, ignorando datos vacíos o que empiecen con ciertos prefijos.
    def send_later(self, tid: int, data: str) -> None:
        if not self.url or not data or data.startswith("YOLO_QR_") or data.startswith("QR_CANDIDATE"):
            return
        self.q.put((tid, data))

    # Función principal del hilo que procesa la cola de datos a enviar al backend
    def _run(self) -> None:
        while not self.stop_event.is_set():
            tid, data = self.q.get()
            if data == "__STOP__":
                break
            self._post(tid, data)

    # Realiza la petición POST al backend con el dato del QR
    def _post(self, tid: int, data: str) -> None:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            self.url,
            data=json.dumps({"gs1Code": data}).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        started = now_ms()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                elapsed = now_ms() - started

                prod_log(f"[DB OK] id={tid} qr={data} status={resp.status} request_ms={elapsed:.1f}")
        except urllib.error.HTTPError as error:
            elapsed = now_ms() - started
            body = error.read().decode("utf-8", errors="replace")[:300]

            prod_log(f"[DB HTTP ERROR] id={tid} qr={data} status={error.code} request_ms={elapsed:.1f} body={body}")
        except Exception as error:
            elapsed = now_ms() - started

            prod_log(f"[DB ERROR] id={tid} qr={data} request_ms={elapsed:.1f} error={error}")


# ============================================================================
# LED RGB DE ESTADO
# ============================================================================

# Funciones para manejar el LED RGB de estado, incluyendo parsing de pines GPIO y colores hexadecimales
def parse_led_pins(value: str) -> Tuple[int, int, int]:
    try:
        parts = [int(item.strip()) for item in value.split(",")]
    except Exception as exc:
        raise RuntimeError("[LED ERROR] --led_pins debe tener formato R,G,B. Ejemplo: 17,27,22") from exc
    if len(parts) != 3:
        raise RuntimeError("[LED ERROR] --led_pins debe tener exactamente 3 GPIO BCM. Ejemplo: 17,27,22")
    return parts[0], parts[1], parts[2]


# Convierte un color hexadecimal en formato #RRGGBB a una tupla de valores RGB normalizados entre 0.0 y 1.0
def parse_hex_color(value: str) -> Tuple[float, float, float]:
    text = str(value).strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise RuntimeError(f"[LED ERROR] Color inválido: {value}. Usa formato #RRGGBB, por ejemplo #00FF00.")
    try:
        r = int(text[0:2], 16) / 255.0
        g = int(text[2:4], 16) / 255.0
        b = int(text[4:6], 16) / 255.0
    except ValueError as exc:
        raise RuntimeError(f"[LED ERROR] Color inválido: {value}. Usa formato #RRGGBB, por ejemplo #00FF00.") from exc
    return r, g, b

# Determina el color del LED RGB basado en la cantidad de códigos QR detectados
def color_for_qr_count(args: argparse.Namespace, count: int) -> Tuple[float, float, float]:
    if count <= 0:
        return parse_hex_color(args.led_color_auth_ok)
    if count == 1:
        return parse_hex_color(args.led_color_qr_1)
    if count == 2:
        return parse_hex_color(args.led_color_qr_2)
    if count == 3:
        return parse_hex_color(args.led_color_qr_3)
    if count == 4:
        return parse_hex_color(args.led_color_qr_4)
    return parse_hex_color(args.led_color_qr_5_plus)

# Clase que maneja el LED RGB de estado
class StatusLed:
    # inicializa el LED RGB basado en los argumento
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.enabled = not args.led_disable
        self.led = None
        self.last_color: Optional[Tuple[float, float, float]] = None
        if not self.enabled:
            return

        red_pin, green_pin, blue_pin = parse_led_pins(args.led_pins)
        try:
            from gpiozero import RGBLED
            self.led = RGBLED(
                red=red_pin,
                green=green_pin,
                blue=blue_pin,
                active_high=not args.led_common_anode,
                pwm=True,
            )
            cprint(args, f"[LED] enabled pins_bcm=R{red_pin},G{green_pin},B{blue_pin} common_anode={args.led_common_anode}", force=True)
        except Exception as error:
            self.enabled = False
            self.led = None
            cprint(args, f"[LED WARN] No se pudo inicializar LED RGB: {error}", force=True)

    # Cambia el color del LED RGB
    def set_color(self, color: Tuple[float, float, float]) -> None:
        if not self.enabled or self.led is None:
            return
        if self.last_color == color:
            return
        try:
            self.led.color = color
            self.last_color = color
        except Exception as error:
            self.enabled = False
            cprint(self.args, f"[LED WARN] No se pudo cambiar color LED: {error}", force=True)

    # Establece el color del LED al iniciar el programa
    def set_startup(self) -> None:
        self.set_color(parse_hex_color(self.args.led_color_startup))

    # Establece el color del LED cuando el login a la API/backend es exitoso
    def set_login_ok(self) -> None:
        self.set_color(parse_hex_color(self.args.led_color_login_ok))

    # Establece el color del LED cuando el sistema ya terminó de cargar y está funcional, listo para detectar códigos QR
    def set_auth_ok(self) -> None:
        self.set_color(parse_hex_color(self.args.led_color_auth_ok))

    # Establece el color del LED basado en la cantidad de códigos QR detectados
    def set_qr_count(self, count: int) -> None:
        self.set_color(color_for_qr_count(self.args, count))

    # Apaga el LED RGB
    def off(self) -> None:
        if not self.enabled or self.led is None:
            return
        try:
            self.led.off()
        except Exception:
            pass


# ============================================================================
# CÁMARA
# ============================================================================

# Función para abrir la fuente de video, ya sea picamera o USB/OpenCv
def open_source(args: argparse.Namespace, width: int, height: int):
    started = now_ms()
    if args.source.startswith("picamera"):
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (width, height), "format": args.picamera_format},
            buffer_count=4,
        )
        cam.configure(config)
        try:
            controls = {"AwbEnable": True, "AeEnable": True, "FrameRate": args.camera_fps}
            if args.exposure_time > 0:
                controls["ExposureTime"] = int(args.exposure_time)
            if args.analogue_gain > 0:
                controls["AnalogueGain"] = float(args.analogue_gain)
            cam.set_controls(controls)
        except Exception as error:
            cprint(args, f"[CAM WARN] controls={error}", force=True)
        cam.start()
        time.sleep(0.3)
        cprint(args, f"[CAM] source=picamera size={width}x{height} open_ms={now_ms() - started:.1f}", force=True)
        return cam, "picamera"

    if args.source.startswith("usb"):
        idx = int(args.source.replace("usb", ""))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("[CAM ERROR] No se pudo abrir la fuente.")
    cprint(args, f"[CAM] source=opencv size={width}x{height} open_ms={now_ms() - started:.1f}", force=True)
    return cap, "opencv"


# Función para leer un frame de la fuente de video
def read_frame(capture, source_type: str, color_fix: str):
    if source_type == "picamera":
        frame = capture.capture_array()
        if frame is None:
            return False, None
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if color_fix == "rgb2bgr":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif color_fix == "bgr2rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True, frame
    ok, frame = capture.read()
    return (ok and frame is not None), frame

# Función para cerrar la fuente de video de forma segura
def close_source(capture, source_type: str) -> None:
    try:
        capture.stop() if source_type == "picamera" else capture.release()
    except Exception:
        pass


# ============================================================================
# DECODIFICACIÓN / DETECCIÓN RÁPIDA
# ============================================================================

# Redimensiona el frame para acelerar la decodificación, devolviendo el frame redimensionado y los factores de escala 
def resize_for_decode(frame: np.ndarray, decode_width: int) -> Tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    if decode_width <= 0 or w <= decode_width:
        return frame, 1.0, 1.0
    scale = decode_width / float(w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, w / float(new_w), h / float(new_h)


# NO USAR-- Función para mejorar la visibilidad de códigos QR en imágenes con poca luz
def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    if frame.ndim == 2:
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(frame)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

# Genera variantes del frame para mejorar la decodificación, incluyendo redimensionado, CLAHE y umbral adaptativo
def make_decode_variants(img: np.ndarray) -> List[np.ndarray]:
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    variants = [gray]
    h, w = gray.shape[:2]
    if w > 0 and h > 0:
        variants.append(cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)
    if w > 0 and h > 0:
        variants.append(cv2.resize(clahe, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    try:
        variants.append(cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 31, 5))
    except Exception:
        pass
    try:
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
    except Exception:
        pass
    return variants

# Convierte un conjunto de puntos a una caja delimitadora
def box_from_points(points: np.ndarray, sx: float, sy: float) -> Box:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

# Función para decodificar códigos QR usando pyzbar  sobre una imagen reducida.
def decode_qrs_pyzbar_fast(frame: np.ndarray, decode_width: int, strong: bool = False) -> List[dict]:
    small, sx, sy = resize_for_decode(frame, decode_width)
    if not strong:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        variants = [gray]
    else:
        variants = make_decode_variants(small)

    results: List[dict] = []
    seen = set()
    for variant in variants:
        for qr in decode(variant, symbols=[ZBarSymbol.QRCODE]):
            data = qr.data.decode("utf-8", errors="replace")
            if not data or data in seen:
                continue
            seen.add(data)
            x, y, w, h = qr.rect
            box = (int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy))
            results.append({
                "data": data,
                "box": box,
                "center": ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                "decoded": True,
                "score": 1.0,
                "source": "pyzbar_strong" if strong else "pyzbar_fast",
            })
    return results


# Funcion para detectar códigos QR usando el detector de OpenCV, devolviendo tanto los resultados decodificados como los candidatos no decodificados.
def detect_qrs_opencv(frame: np.ndarray, decode_width: int, qr_detector) -> Tuple[List[dict], List[dict]]:
    """Retorna (decoded_results, presence_candidates)."""
    small, sx, sy = resize_for_decode(frame, decode_width)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    decoded_results: List[dict] = []
    candidates: List[dict] = []
    try:
        ok, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)
    except Exception:
        ok, decoded_info, points = False, [], None

    if not ok or points is None:
        try:
            found, points = qr_detector.detectMulti(gray)
        except Exception:
            found, points = False, None
        if not found or points is None:
            return decoded_results, candidates
        decoded_info = [""] * len(points)

    seen = set()
    for idx, pts in enumerate(points):
        box = box_from_points(pts, sx, sy)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        text = ""
        if idx < len(decoded_info) and decoded_info[idx]:
            text = str(decoded_info[idx])
        if text and text not in seen:
            seen.add(text)
            decoded_results.append({
                "data": text,
                "box": box,
                "center": center,
                "decoded": True,
                "score": 1.0,
                "source": "opencv_qr",
            })
        else:
            candidates.append({
                "data": f"QR_CANDIDATE_{idx + 1}",
                "box": box,
                "center": center,
                "decoded": False,
                "score": 0.50,
                "source": "opencv_candidate",
            })
    return decoded_results, candidates

# Función para decodificar un recorte de imagen que contiene un código QR
def decode_crop(crop: np.ndarray, strong: bool = False) -> Optional[str]:
    if crop is None or crop.size == 0:
        return None
    variants = make_decode_variants(crop) if strong else [
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ]
    for variant in variants:
        decoded = decode(variant, symbols=[ZBarSymbol.QRCODE])
        if decoded:
            return decoded[0].data.decode("utf-8", errors="replace")
    return None

# Función para ajustar una caja delimitadora dentro de los límites de la imagen, con un padding opcional
def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int, pad: int = 0) -> Box:
    return (
        max(0, int(x1) - pad),
        max(0, int(y1) - pad),
        min(width - 1, int(x2) + pad),
        min(height - 1, int(y2) + pad),
    )


# ============================================================================
# YOLO
# ============================================================================

# Carga el modelo YOLO con ultralytics
def load_yolo_model(args: argparse.Namespace):
    if not args.model:
        raise RuntimeError("[MODEL ERROR] Debes indicar --model con detector=yolo.")
    from ultralytics import YOLO
    started = now_ms()
    try:
        model = YOLO(args.model, task="detect")
    except TypeError:
        model = YOLO(args.model)
    cprint(args, f"[MODEL] loaded path={args.model} load_ms={now_ms() - started:.1f}", force=True)
    return model

# Función de warmup para YOLO, ejecuta inferencias rápidas con una imagen vacía para estabilizar tiempos
def warmup_yolo(model, args: argparse.Namespace, width: int, height: int) -> None:
    if model is None or args.warmup_runs <= 0:
        return
    warm_img = np.zeros((height, width, 3), dtype=np.uint8)
    kwargs = {"conf": args.conf, "iou": args.iou, "max_det": args.max_det, "verbose": False}
    if args.yolo_imgsz > 0:
        kwargs["imgsz"] = args.yolo_imgsz
    started = now_ms()
    for _ in range(args.warmup_runs):
        model.predict(warm_img, **kwargs)
    cprint(args, f"[MODEL] warmup_runs={args.warmup_runs} warmup_ms={now_ms() - started:.1f}", force=True)

# Función para determinar si una caja detectada por YOLO es razonable para ser considerada como candidata a código QR
# basada en su tamaño, relación de aspecto y puntaje de confianza.
def yolo_box_is_reasonable(box: Box, score: float, frame_w: int, frame_h: int, args: argparse.Namespace) -> bool:
    x1, y1, x2, y2 = box
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw < args.yolo_min_size or bh < args.yolo_min_size:
        return False
    area_ratio = (bw * bh) / float(frame_w * frame_h)
    if area_ratio < args.yolo_min_area_ratio or area_ratio > args.yolo_max_area_ratio:
        return False
    aspect = max(bw / max(1, bh), bh / max(1, bw))
    if aspect > args.yolo_max_aspect:
        return False
    if score < args.conf:
        return False
    return True

# Función para detectar códigos QR usando un modelo YOLO
def detect_qrs_yolo(frame: np.ndarray, model, args: argparse.Namespace, path: str = "yolo_crop") -> Tuple[List[dict], ScanMetrics]:
    metrics = ScanMetrics(path=path)

    # Si estamos en servicio con --headless --quiet, no calculamos tiempos internos de YOLO.
    use_metrics = metrics_enabled(args)
    scan_start = now_ms() if use_metrics else 0.0
    h, w = frame.shape[:2]

    prep_start = now_ms() if use_metrics else 0.0
    infer_frame = enhance_low_light(frame) if args.low_light_enhance else frame
    if use_metrics:
        metrics.preprocess_ms = now_ms() - prep_start

    kwargs = {"conf": args.conf, "iou": args.iou, "max_det": args.max_det, "verbose": False}
    if args.yolo_imgsz > 0:
        kwargs["imgsz"] = args.yolo_imgsz

    yolo_start = now_ms() if use_metrics else 0.0
    results = model.predict(infer_frame, **kwargs)
    if use_metrics:
        metrics.yolo_wall_ms = now_ms() - yolo_start

    raw_candidates = []
    for result in results:
        if use_metrics:
            speed = getattr(result, "speed", {}) or {}
            metrics.yolo_reported_infer_ms += float(speed.get("inference", 0.0) or 0.0)
        if result.boxes is None:
            continue
        for detected_box in result.boxes:
            raw_x1, raw_y1, raw_x2, raw_y2 = detected_box.xyxy[0].cpu().numpy()
            score = float(detected_box.conf[0])
            box = clamp_box(raw_x1, raw_y1, raw_x2, raw_y2, w, h, pad=0)
            if yolo_box_is_reasonable(box, score, w, h, args):
                raw_candidates.append((score, raw_x1, raw_y1, raw_x2, raw_y2, box))
            else:
                if use_metrics:
                    metrics.rejected_count += 1

    raw_candidates.sort(key=lambda item: item[0], reverse=True)
    detections: List[dict] = []
    decode_limit = max(0, int(args.yolo_decode_topk))

    for idx, (score, raw_x1, raw_y1, raw_x2, raw_y2, box) in enumerate(raw_candidates):
        x1, y1, x2, y2 = box
        crop_box = clamp_box(raw_x1, raw_y1, raw_x2, raw_y2, w, h, pad=args.yolo_pad)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        qr_text: Optional[str] = None
        metrics.candidate_count += 1
        if use_metrics:
            metrics.confidences.append(score)

        if args.yolo_decode_crop and idx < decode_limit:
            if use_metrics:
                metrics.crop_count += 1
                crop_start = now_ms()
            else:
                crop_start = 0.0
            px1, py1, px2, py2 = crop_box
            crop = frame[py1:py2, px1:px2]
            qr_text = decode_crop(crop, strong=args.strong_crop_decode)
            if not qr_text and args.low_light_enhance:
                qr_text = decode_crop(infer_frame[py1:py2, px1:px2], strong=True)
            if use_metrics:
                metrics.crop_decode_ms += now_ms() - crop_start

        decoded_ok = bool(qr_text)
        if decoded_ok:
            metrics.decoded_count += 1
        detections.append({
            "data": qr_text if qr_text else f"YOLO_QR_{score:.2f}",
            "box": box,
            "crop_box": crop_box,
            "center": (cx, cy),
            "decoded": decoded_ok,
            "score": score,
            "source": "yolo_crop" if decoded_ok else "yolo_candidate",
        })

    if use_metrics:
        metrics.total_ms = now_ms() - scan_start
    return detections, metrics

# Función para fusionar las métricas de dos escaneos
def merge_metrics(base: ScanMetrics, yolo: ScanMetrics, path: str) -> ScanMetrics:
    base.path = path
    base.total_ms += yolo.total_ms
    base.preprocess_ms += yolo.preprocess_ms
    base.yolo_wall_ms += yolo.yolo_wall_ms
    base.yolo_reported_infer_ms += yolo.yolo_reported_infer_ms
    base.crop_decode_ms += yolo.crop_decode_ms
    base.crop_count += yolo.crop_count
    base.candidate_count += yolo.candidate_count
    base.decoded_count += yolo.decoded_count
    base.rejected_count += yolo.rejected_count
    base.confidences.extend(yolo.confidences)
    return base

# Función principal para detectar códigos QR en un frame 
# utilizando una estrategia de detección primaria (pyzbar u OpenCV) y opcionalmente YOLO como respaldo
def detect_qrs(frame: np.ndarray, args: argparse.Namespace, model, qr_detector) -> Tuple[List[dict], ScanMetrics]:
    # En --quiet  no se calcula tiempos de cada detector
    use_metrics = metrics_enabled(args)
    scan_start = now_ms() if use_metrics else 0.0
    metrics = ScanMetrics(path="-")

    # Si se configuró YOLO como único detector o con modo de fallback "siempre", se omiten los detectores primarios y se va directo a YOLO.
    if args.detector == "yolo" and (args.fast_yolo or args.yolo_no_pyzbar_first):
        return detect_qrs_yolo(frame, model, args, path="yolo_direct")

    decoded_results: List[dict] = []
    presence_candidates: List[dict] = []

    # Primero se intenta detectar y decodificar con pyzbar  (si está configurado como primario)
    if args.primary_detector in ("pyzbar", "hybrid"):
        start = now_ms() if use_metrics else 0.0
        decoded_results = decode_qrs_pyzbar_fast(frame, args.decode_width, strong=args.full_strong_decode)
        if use_metrics:
            metrics.pyzbar_ms = now_ms() - start
        if decoded_results:
            metrics.path = "pyzbar_first"
            if use_metrics:
                metrics.total_ms = now_ms() - scan_start
                metrics.confidences = [1.0] * len(decoded_results)
            metrics.candidate_count = len(decoded_results)
            metrics.decoded_count = len(decoded_results)
            return decoded_results, metrics

    # Si pyzbar no encontró nada y se configuró OpenCV como primario o híbrido, se intenta con el detector de OpenCV
    if args.primary_detector in ("opencv", "hybrid"):
        start = now_ms() if use_metrics else 0.0
        cv_decoded, cv_candidates = detect_qrs_opencv(frame, args.decode_width, qr_detector)
        if use_metrics:
            metrics.opencv_ms = now_ms() - start
        if cv_decoded:
            metrics.path = "opencv_qr"
            if use_metrics:
                metrics.total_ms = now_ms() - scan_start
                metrics.confidences = [1.0] * len(cv_decoded)
            metrics.candidate_count = len(cv_decoded)
            metrics.decoded_count = len(cv_decoded)
            return cv_decoded, metrics
        presence_candidates = cv_candidates

    # Si se configuró pyzbar como detector primario pero no se encontró ningún QR, se devuelve la lista de candidatos de presencia sin intentar YOLO
    if args.detector == "pyzbar":
        metrics.path = "primary_no_decode"
        if use_metrics:
            metrics.total_ms = now_ms() - scan_start
        metrics.candidate_count = len(presence_candidates)
        metrics.decoded_count = 0
        return presence_candidates, metrics

    # Determinar si se debe usar YOLO como respaldo según la configuración y la presencia de candidatos de OpenCV
    use_yolo = False
    if args.yolo_fallback_mode == "always":
        use_yolo = True
    elif args.yolo_fallback_mode == "presence":
        use_yolo = len(presence_candidates) > 0
    elif args.yolo_fallback_mode == "never":
        use_yolo = False

    # Si se decidió usar YOLO como respaldo, se ejecuta la detección con YOLO y se fusionan las métricas
    if use_yolo:
        yolo_dets, yolo_metrics = detect_qrs_yolo(frame, model, args, path="yolo_after_primary_fail")
        merged = merge_metrics(metrics, yolo_metrics, path="yolo_after_primary_fail")
        if use_metrics:
            merged.total_ms = now_ms() - scan_start
        # Si YOLO no devolvió nada útil, conserva candidatos OpenCV para tracking visual.
        return (yolo_dets if yolo_dets else presence_candidates), merged

    # Si no se usó YOLO, se devuelve la lista de candidatos de presencia detectados por OpenCV o una lista vacia
    metrics.path = "primary_no_qr" if not presence_candidates else "opencv_candidate_no_yolo"
    if use_metrics:
        metrics.total_ms = now_ms() - scan_start
    metrics.candidate_count = len(presence_candidates)
    return presence_candidates, metrics


# ============================================================================
# TRACKER
# ============================================================================

def distance(p1: Point, p2: Point) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.sqrt(dx * dx + dy * dy))

# Trackea los QR detectados a lo largo del tiempo, asignando IDs persistentes y manejando estados de pérdida y reaparición.
class QRTracker:
    # Mantiene ids persistentes
    # params:
    #   max_lost: cantidad máxima de frames que un QR puede estar "perdido" antes de considerarlo expirado y eliminarlo.
    #   trail_seconds: cantidad de segundos para mantener el rastro histórico de posiciones (trail)
    #   trail_len: cantidad máxima de puntos en el trail histórico
    #   match_dist: distancia máxima en píxeles para considerar que una detección nueva
    def __init__(self, max_lost: int = 150, trail_seconds: float = 30.0, trail_len: int = 60, match_dist: float = 200.0) -> None:
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1
        self.max_lost = int(max_lost)
        self.trail_seconds = float(trail_seconds)
        self.trail_len = int(trail_len)
        self.match_dist = float(match_dist)

    # Actualiza el estado del tracker con las nuevas detecciones. Retorna una lista de tracks expirados
    def update(self, detections: List[dict], lost_increment: int = 1) -> List[dict]:
        now = time.time()
        # Marcar todos los tracks como no actualizados inicialmente
        for tr in self.tracks.values():
            tr["updated"] = False

        # Para cada detección, intentar encontrar un track coincidente por ID. Si no se encuentra coincidencia, crear un nuevo track
        for det in detections:
            center = det.get("center")
            box = det.get("box")
            if center is None or box is None:
                continue
            data = str(det.get("data", "")).strip()
            decoded = bool(det.get("decoded", False)) and data and not data.startswith("YOLO_QR_") and not data.startswith("QR_CANDIDATE")
            score = float(det.get("score", 0.0))
            source = det.get("source", "qr")

            # Primero intentar hacer match por contenido decodificado
            matched_id = None
            if decoded:
                for tid, tr in self.tracks.items():
                    if tr.get("decoded", False) and tr.get("data") == data:
                        matched_id = tid
                        break

            # if matched_id is None:
            #     best_id = None
            #     best_dist = 1e9
            #     for tid, tr in self.tracks.items():
            #         d = distance(center, tr["center"])
            #         if d < best_dist and d < self.match_dist:
            #             best_dist = d
            #             best_id = tid
            #     matched_id = best_id

            # Si no se encontró ningún match por contenido decodificado, intentar hacer match por proximidad
            if matched_id is None:
                best_id = None
                best_dist = 1e9

                for tid, tr in self.tracks.items():
                    # Si el QR actual está decodificado y el track viejo también,
                    # NO permitir match por distancia si el contenido es diferente.
                    if decoded and tr.get("decoded", False):
                        if tr.get("data") != data:
                            continue

                    # Si el track viejo tiene un QR decodificado, NO permitir match por distancia si el nuevo QR no está decodificado (evitar que un candidato sin data reemplace a un track con data).
                    d = distance(center, tr["center"])
                    if d < best_dist and d < self.match_dist:
                        best_dist = d
                        best_id = tid

                matched_id = best_id

            # Si no se encontró ningún track coincidente, crear uno nuevo con un ID único
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                self.tracks[matched_id] = {
                    "id": matched_id,
                    "data": data,
                    "box": box,
                    "center": center,
                    "movement": 0.0,
                    "speed_px_s": 0.0,
                    "dx": 0,
                    "dy": 0,
                    "last_time": now,
                    "lost": 0,
                    "updated": True,
                    "trail": [(now, center)],
                    "decoded": decoded,
                    "score": score,
                    "source": source,
                }
                continue

            # Si se encontró un track coincidente, actualizar su información con la nueva detección y calcular movimiento y velocidad
            tr = self.tracks[matched_id]
            dt = max(now - tr["last_time"], 0.0001)
            dx = center[0] - tr["center"][0]
            dy = center[1] - tr["center"][1]
            move = float(np.sqrt(dx * dx + dy * dy))
            tr.update({
                "box": box,
                "center": center,
                "movement": move,
                "speed_px_s": move / dt,
                "dx": dx,
                "dy": dy,
                "last_time": now,
                "lost": 0,
                "updated": True,
                "score": score,
                "source": source,
            })
            # Si el track no tenía data decodificada antes, o si la nueva detección tiene data decodificada, actualizar el campo data del track.
            if decoded or not tr.get("decoded", False):
                tr["data"] = data
                tr["decoded"] = decoded
            tr["trail"].append((now, center))
            threshold = now - self.trail_seconds
            tr["trail"] = [item for item in tr["trail"] if item[0] >= threshold][-self.trail_len:]

        # Incrementar el contador de "lost" para los tracks que no fueron actualizados en esta ronda, y eliminar los tracks que superen el umbral de pérdida máxima. Retornar la lista de tracks expirados.
        expired: List[dict] = []
        for tid, tr in list(self.tracks.items()):
            if not tr.get("updated", False):
                tr["lost"] += max(1, int(lost_increment))
            if tr["lost"] > self.max_lost:
                expired.append({"id": tid, "data": tr.get("data", "")})
                del self.tracks[tid]
        return expired

    # Retorna un diccionario de los tracks que están actualmente visibles (no perdidos)
    def visible_tracks(self) -> Dict[int, dict]:
        return {tid: tr for tid, tr in self.tracks.items() if tr["lost"] <= self.max_lost}

    # Retorna la cantidad de tracks visibles (no perdidos)
    def visible_count(self) -> int:
        return sum(1 for tr in self.tracks.values() if tr.get("lost", 0) == 0)

    # Retorna la cantidad de tracks que están en estado de gracia (perdidos pero no aún expirados)
    def grace_count(self) -> int:
        return sum(1 for tr in self.tracks.values() if tr.get("lost", 0) > 0)


# Función para convertir los desplazamientos dx, dy en una descripción textual de la dirección del movimiento (ej. "arriba-izquierda", "derecha", "quieto", etc.)
def direction_text(dx: float, dy: float) -> str:
    if abs(dx) < 3 and abs(dy) < 3:
        return "quieto"
    horizontal = "derecha" if dx > 3 else "izquierda" if dx < -3 else ""
    vertical = "abajo" if dy > 3 else "arriba" if dy < -3 else ""
    return f"{vertical}-{horizontal}" if horizontal and vertical else horizontal or vertical

# Función para dibujar los tracks actuales sobre el frame, mostrando cajas delimitadoras, centros, información de movimiento y otros datos relevantes. 
def draw_tracks(frame: np.ndarray, tracks: Dict[int, dict], draw_trail: bool) -> None:
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = tr["box"]
        cx, cy = tr["center"]
        color = (0, 255, 0) if tr.get("lost", 0) == 0 else (0, 180, 255)
        if draw_trail:
            pts = [point for _, point in tr.get("trail", [])]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        data = str(tr.get("data", ""))[:30]
        label1 = f"ID {tid} | {data}"
        label2 = f"Move {tr.get('movement', 0.0):.1f}px | {tr.get('speed_px_s', 0.0):.1f}px/s | {direction_text(tr.get('dx', 0), tr.get('dy', 0))}"
        label2 += f" | conf {tr.get('score', 0.0):.2f} | {tr.get('source', '')}"
        cv2.putText(frame, label1, (x1, max(y1 - 35, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, label2, (x1, max(y1 - 12, 45)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

# Función para dibujar las cajas de los candidatos detectados por YOLO que no fueron decodificados
def draw_yolo_candidates(frame: np.ndarray, detections: List[dict]) -> None:
    for det in detections:
        if det.get("decoded", False):
            continue
        if not str(det.get("source", "")).startswith("yolo"):
            continue
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.putText(frame, f"YOLO {det.get('score', 0.0):.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

# Función para dibujar un panel de información (HUD) sobre el frame
def draw_hud(frame: np.ndarray, args: argparse.Namespace, tracker: QRTracker, metrics: Metrics) -> None:
    last = metrics.last_scan
    lines = [
        f"model={safe_model_name(args.model)} detector={args.detector.upper()} primary={args.primary_detector} fallback={args.yolo_fallback_mode}",
        f"FPS camera={metrics.fps():.1f} scan={metrics.scan_fps():.2f} scan_every={args.scan_every}",
        f"path={last.path} pyzbar={last.pyzbar_ms:.1f}ms opencv={last.opencv_ms:.1f}ms yolo={last.yolo_wall_ms:.1f}ms crop={last.crop_decode_ms:.1f}ms",
        f"cand={last.candidate_count} decoded={last.decoded_count} rejected={last.rejected_count} visible={tracker.visible_count()} grace={tracker.grace_count()}",
        "Q: salir",
    ]
    panel_h = 22 + len(lines) * 22
    panel_w = min(frame.shape[1], 980)
    panel = frame[:panel_h, :panel_w]
    black = np.zeros_like(panel)
    cv2.addWeighted(black, 0.62, panel, 0.38, 0.0, panel)
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()
    if args.show:
        args.view = "camera"
    if args.log_only or args.headless:
        args.view = "logs"
    if args.no_console_logs:
        args.quiet = True
    if args.fast_yolo:
        args.yolo_no_pyzbar_first = True

    width, height = parse_resolution(args.resolution)
    metrics = Metrics(args.metrics_window)
    capture = None
    source_type = ""
    backend = None
    status_led = None
    qr_detector = cv2.QRCodeDetector()

    cprint(args, "[INIT] QR detector con métricas inicializando.", force=True)
    cprint(args, f"[INIT] detector={args.detector} model={safe_model_name(args.model)} resolution={width}x{height} view={args.view}", force=True)
    cprint(args, (
        f"[INIT] primary={args.primary_detector} fallback={args.yolo_fallback_mode} imgsz={args.yolo_imgsz} "
        f"scan_every={args.scan_every} decode_width={args.decode_width} conf={args.conf:.2f} max_det={args.max_det} "
        f"decode_topk={args.yolo_decode_topk} quiet={args.quiet} full_strong_decode={args.full_strong_decode} "
        f"strong_crop_decode={args.strong_crop_decode}"
    ), force=True)

    status_led = StatusLed(args)
    status_led.set_startup()

    try:
        # peticion de logeo al backend
        token = login_camera(args) if args.backend_url else None
        backend = BackendWorker(args, token, metrics)
        backend.start()
        if args.backend_url:
            status_led.set_login_ok()

        # [1] Modelo yolo (si aplica), se hace warmup antes de abrir la cámara para no afectar los tiempos de los primeros frames.
        model = None
        if args.detector == "yolo":
            model = load_yolo_model(args)
            warmup_yolo(model, args, width, height)
        status_led.set_auth_ok()

        # [2] Abrir fuente de video
        capture, source_type = open_source(args, width, height)
        # [3] Crear tracker para seguimiento de QR codes
        tracker = QRTracker(args.max_lost, args.trail_seconds, args.trail_len, args.track_match_dist)

        frame_count = 0
        last_detections: List[dict] = []
        seen_qr_codes: set[str] = set()

        cprint(args, f"[START] running source={args.source} camera_fps_requested={args.camera_fps}", force=True)
        cprint(args, "[START] Recomendado: primary=hybrid + fallback=presence. Q termina.", force=True)


        while True:
            frame_count += 1
            ok, frame = read_frame(capture, source_type, args.color_fix)
            if not ok or frame is None:
                prod_log("[CAM ERROR] Lectura de frame fallida.")
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Determinar si se debe realizar un escaneo de detección en este frame según la configuración de scan_every
            should_scan = frame_count % max(1, args.scan_every) == 0
            if should_scan:
                last_detections, scan_metrics = detect_qrs(frame, args, model, qr_detector)

                metrics.record_scan(scan_metrics)
            
            # Actualizar el tracker con las nuevas detecciones (si se hizo un escaneo) o incrementar el contador de pérdida para los tracks existentes si no se hizo escaneo 
            expired = tracker.update(last_detections if should_scan else [], lost_increment=args.scan_every if should_scan else 1)
            for item in expired:
                pass

            # Actualizar el LED de estado con la cantidad de QR visibles y en gracia
            status_led.set_qr_count(tracker.visible_count() + tracker.grace_count())

            # Imprimir en consola los QR codes nuevos que aparecieron en el tracker, y enviar al backend si corresponde
            for tid, tr in tracker.visible_tracks().items():
                data = str(tr.get("data", "")).strip()
                decoded = bool(tr.get("decoded", False))
                # Solo considerar como "nuevo" un QR que no esté perdido, que tenga data decodificada y que no haya sido visto antes
                if (
                    tr.get("lost", 0) == 0
                    and decoded
                    and data
                    and not data.startswith("YOLO_QR_")
                    and not data.startswith("QR_CANDIDATE")
                    and data not in seen_qr_codes
                ):
                    seen_qr_codes.add(data)
                    if backend:
                        backend.send_later(tid, data)

            metrics.record_frame()

            # Si la vista es "camera", dibujar los tracks y candidatos sobre el frame 
            if args.view == "camera":
                display = frame.copy()
                draw_tracks(display, tracker.visible_tracks(), args.draw_trail)
                if args.draw_yolo_candidates:
                    draw_yolo_candidates(display, last_detections)
                if args.metrics_overlay:
                    draw_hud(display, args, tracker, metrics)
                cv2.imshow("QR Tracking | HYBRID FAST FALLBACK", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    cprint(args, "[STOP] key=q", force=True)
                    break
    except KeyboardInterrupt:
        prod_log("[STOP] keyboard_interrupt")
    finally:
        cprint(args, "[STOP] Cerrando recursos.", force=True)
        if status_led is not None:
            status_led.off()
        if backend is not None:
            backend.stop()
        if capture is not None:
            close_source(capture, source_type)
        if args.view == "camera":
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        cprint(args, "[STOP] Finalizado.", force=True)


if __name__ == "__main__":
    main()