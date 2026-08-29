#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Análisis Deportivo - TheStats API v1.0.0
Autor: Backend Senior
Entorno: GitHub Actions / Cloud
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set

import requests
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────

BASE_URL = "https://api.thestatsapi.com/api"
API_KEY = os.environ.get("THESTATS_API_KEY", "")
if not API_KEY:
    logging.error("THESTATS_API_KEY no está configurada en las variables de entorno.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

NOMBRES_LIGAS_ELITE = [
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "UEFA Champions League",
    "UEFA Europa League",
    "Copa Libertadores"
]

MAX_PARTIDOS_DIA = 10
MAX_PARTIDOS_HISTORICO = 10
MAX_FIXTURES_POR_EQUIPO = 100
MIN_TASA_SEGURA = 0.80

UMBRALES_CORNERS = [2.5, 3.5, 4.5, 5.5, 6.5]
UMBRALES_REMATES_ARCO = [2.5, 3.5, 4.5, 5.5]
UMBRALES_GOLES_TOTALES = [1.5, 2.5, 3.5]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ─────────────────────────────────────────────────────────────
# CAPA DE COMUNICACIÓN HTTP
# ─────────────────────────────────────────────────────────────

def api_get(endpoint: str, params: Dict[str, Any] = None, max_retries: int = 3) -> Dict[str, Any]:
    """
    Ejecuta una petición GET autenticada a TheStats API con reintentos
    exponenciales y manejo de errores.
    """
    url = f"{BASE_URL}{endpoint}"
    for intento in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                espera = 2 ** intento
                logging.warning(f"Rate limit alcanzado. Esperando {espera}s...")
                time.sleep(espera)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logging.error(f"Error de API en {url}: {data['error']}")
            return data
        except requests.exceptions.RequestException as exc:
            logging.warning(f"Intento {intento}/{max_retries} fallido en {url}: {exc}")
            if intento == max_retries:
                logging.error(f"Agotados reintentos para {url}")
                raise
            time.sleep(1.5 ** intento)
    return {}


def api_get_all_pages(endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Itera automáticamente todas las páginas de una respuesta paginada
    y devuelve el array completo de objetos 'data'.
    """
    todos = []
    pagina = 1
    while True:
        params_pag = {**params, "page": pagina, "per_page": 100}
        data = api_get(endpoint, params_pag)
        items = data.get("data", [])
        if not items:
            break
        todos.extend(items)
        meta = data.get("meta", {})
        if pagina >= meta.get("total_pages", 1):
            break
        pagina += 1
        time.sleep(0.4)
    return todos


# ─────────────────────────────────────────────────────────────
# RESOLUCIÓN DINÁMICA DE IDs DE COMPETICIONES
# ─────────────────────────────────────────────────────────────

def resolver_ids_competiciones() -> Set[str]:
    """
    Consulta /football/competitions y resuelve dinámicamente los IDs
    de las ligas élite por nombre exacto o parcial.
    """
    logging.info("Resolviendo IDs de competiciones élite...")
    competiciones = api_get_all_pages("/football/competitions", {})
    ids = set()
    mapeo_nombres = {c["name"].lower(): c["id"] for c in competiciones}
    for nombre in NOMBRES_LIGAS_ELITE:
        nombre_lower = nombre.lower()
        # Búsqueda exacta primero
        if nombre_lower in mapeo_nombres:
            ids.add(mapeo_nombres[nombre_lower])
            continue
        # Búsqueda parcial como fallback
        for comp in competiciones:
            if nombre_lower in comp["name"].lower():
                ids.add(comp["id"])
                break
    logging.info(f"Competiciones resueltas: {len(ids)} / {len(NOMBRES_LIGAS_ELITE)}")
    return ids


# ─────────────────────────────────────────────────────────────
# EXTRACTORES DE DATOS REALES
# ─────────────────────────────────────────────────────────────

def obtener_partidos_del_dia(fecha_str: str, ids_ligas: Set[str]) -> List[Dict[str, Any]]:
    """
    Consulta /football/matches?date_from={fecha}&date_to={fecha}&status=scheduled
    y filtra únicamente partidos pertenecientes a las ligas élite resueltas.
    """
    partidos = api_get_all_pages("/football/matches", {
        "date_from": fecha_str,
        "date_to": fecha_str,
        "status": "scheduled"
    })
    filtrados = []
    for p in partidos:
        comp_id = p.get("competition_id")
        if comp_id in ids_ligas:
            filtrados.append(p)
    return filtrados[:MAX_PARTIDOS_DIA]


def obtener_ultimos_partidos_equipo(team_id: str, es_local: bool, fecha_hasta: str) -> List[Dict[str, Any]]:
    """
    Consulta /football/matches?team_id={team_id}&status=finished&date_to={fecha_hasta}.
    Filtra exactamente los últimos 10 partidos finalizados donde el equipo jugó
    en casa (es_local=True) o fuera (es_local=False).
    """
    partidos = api_get_all_pages("/football/matches", {
        "team_id": team_id,
        "status": "finished",
        "date_to": fecha_hasta
    })
    # Ordenar por fecha descendente (más reciente primero)
    partidos_ordenados = sorted(
        partidos,
        key=lambda x: x.get("utc_date", ""),
        reverse=True
    )
    filtrados = []
    for p in partidos_ordenados:
        home_id = p.get("home_team", {}).get("id")
        away_id = p.get("away_team", {}).get("id")
        if es_local and home_id == team_id:
            filtrados.append(p)
        elif not es_local and away_id == team_id:
            filtrados.append(p)
    return filtrados[:MAX_PARTIDOS_HISTORICO]


def obtener_estadisticas_partido(match_id: str) -> Dict[str, Any]:
    """
    Consulta /football/matches/{match_id}/stats.
    Devuelve el nodo 'data' completo con las estadísticas del partido.
    """
    data = api_get(f"/football/matches/{match_id}/stats")
    return data.get("data", {})


def extraer_stat_equipo(stats_data: Dict[str, Any], team_id: str, match_obj: Dict[str, Any], stat_key: str) -> float:
    """
    Extrae un valor numérico de estadísticas de partido para un equipo específico.
    Dado que las stats vienen en pares home/away, determina si el equipo
    jugó como local o visitante en ese partido y extrae el valor correcto.
    """
    if not stats_data or not match_obj:
        return 0.0

    home_team_id = match_obj.get("home_team", {}).get("id")
    away_team_id = match_obj.get("away_team", {}).get("id")

    overview = stats_data.get("overview", {})
    stat_node = overview.get(stat_key, {})
    all_vals = stat_node.get("all", {})

    if team_id == home_team_id:
        raw = all_vals.get("home")
    elif team_id == away_team_id:
        raw = all_vals.get("away")
    else:
        return 0.0

    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


# ─────────────────────────────────────────────────────────────
# MOTOR MATEMÁTICO DE ANÁLISIS
# ─────────────────────────────────────────────────────────────

def calcular_frecuencia(array_valores: List[float], umbral: float) -> float:
    """
    Calcula la tasa de cumplimiento de un umbral sobre un array de valores reales.
    """
    if not array_valores:
        return 0.0
    cumplen = sum(1 for v in array_valores if v > umbral)
    return (cumplen / len(array_valores)) * 100.0


def evaluar_lineas_seguras(valores: List[float], umbrales: List[float]) -> List[Dict[str, Any]]:
    """
    Evalúa cada umbral. Si la tasa de cumplimiento es >= 80%, se cataloga como 'Segura'.
    """
    lineas = []
    for umbral in umbrales:
        tasa = calcular_frecuencia(valores, umbral)
        lineas.append({
            "linea": umbral,
            "probabilidad": round(tasa, 2),
            "segura": tasa >= (MIN_TASA_SEGURA * 100)
        })
    return lineas


def analizar_mercado_goles_y_btts(partidos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extrae goles totales y BTTS de los matches ya obtenidos (sin llamada adicional).
    """
    goles_totales = []
    btts_si = 0
    for p in partidos:
        score = p.get("score", {}) or {}
        home_goals = score.get("home")
        away_goals = score.get("away")
        # Para partidos finished, score debe tener valores numéricos
        if home_goals is None or away_goals is None:
            continue
        total = home_goals + away_goals
        goles_totales.append(float(total))
        if home_goals > 0 and away_goals > 0:
            btts_si += 1

    total_partidos = len(goles_totales) or 1
    lineas_goles = evaluar_lineas_seguras(goles_totales, UMBRALES_GOLES_TOTALES)

    return {
        "promedio_goles": round(np.mean(goles_totales), 2) if goles_totales else 0.0,
        "lineas_goles": lineas_goles,
        "btts_frecuencia": round((btts_si / total_partidos) * 100, 2),
        "btts_seguro": (btts_si / total_partidos) >= MIN_TASA_SEGURA
    }


# ─────────────────────────────────────────────────────────────
# ORQUESTADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────

def procesar_partido(match_obj: Dict[str, Any], ids_ligas: Set[str]) -> Optional[Dict[str, Any]]:
    """
    Procesa un único partido del día: extrae historial condicionado,
    estadísticas de rendimiento, goles, BTTS y genera el nodo JSON.
    """
    match_id = match_obj.get("id")
    fecha = match_obj.get("utc_date", "")[:10]
    liga_id = match_obj.get("competition_id")
    season_id = match_obj.get("season_id")
    home_team = match_obj.get("home_team", {})
    away_team = match_obj.get("away_team", {})
    home_id = home_team.get("id")
    away_id = away_team.get("id")

    logging.info(f"Procesando partido {match_id}: {home_team.get('name')} vs {away_team.get('name')}")

    # ── Extracción de historial condicionado ──
    # Usamos la fecha del partido como límite superior para no incluir partidos futuros
    fecha_hasta = match_obj.get("utc_date", datetime.now(timezone.utc).isoformat())[:10]

    partidos_local_casa = obtener_ultimos_partidos_equipo(home_id, es_local=True, fecha_hasta=fecha_hasta)
    partidos_visitante_fuera = obtener_ultimos_partidos_equipo(away_id, es_local=False, fecha_hasta=fecha_hasta)

    if len(partidos_local_casa) < 5 or len(partidos_visitante_fuera) < 5:
        logging.warning(f"Datos insuficientes para {match_id}. Se requieren mínimo 5 partidos históricos por bando.")
        return None

    # ── Recolección de estadísticas de rendimiento ──
    corners_local = []
    remates_arco_local = []
    tarjetas_local = []

    for p in partidos_local_casa:
        mid = p.get("id")
        stats = obtener_estadisticas_partido(mid)
        corners_local.append(extraer_stat_equipo(stats, home_id, p, "corner_kicks"))
        remates_arco_local.append(extraer_stat_equipo(stats, home_id, p, "shots_on_target"))
        tarjetas_local.append(extraer_stat_equipo(stats, home_id, p, "yellow_cards"))
        time.sleep(0.3)

    corners_visitante = []
    remates_arco_visitante = []
    tarjetas_visitante = []

    for p in partidos_visitante_fuera:
        mid = p.get("id")
        stats = obtener_estadisticas_partido(mid)
        corners_visitante.append(extraer_stat_equipo(stats, away_id, p, "corner_kicks"))
        remates_arco_visitante.append(extraer_stat_equipo(stats, away_id, p, "shots_on_target"))
        tarjetas_visitante.append(extraer_stat_equipo(stats, away_id, p, "yellow_cards"))
        time.sleep(0.3)

    # ── Cálculo de promedios reales ──
    promedio_corners_local = round(np.mean(corners_local), 2) if corners_local else 0.0
    promedio_corners_visitante = round(np.mean(corners_visitante), 2) if corners_visitante else 0.0
    promedio_corners_total = round(np.mean(corners_local + corners_visitante), 2)

    promedio_remates_local = round(np.mean(remates_arco_local), 2) if remates_arco_local else 0.0
    promedio_remates_visitante = round(np.mean(remates_arco_visitante), 2) if remates_arco_visitante else 0.0
    promedio_remates_total = round(np.mean(remates_arco_local + remates_arco_visitante), 2)

    promedio_tarjetas_local = round(np.mean(tarjetas_local), 2) if tarjetas_local else 0.0
    promedio_tarjetas_visitante = round(np.mean(tarjetas_visitante), 2) if tarjetas_visitante else 0.0
    promedio_tarjetas_total = round(np.mean(tarjetas_local + tarjetas_visitante), 2)

    # ── Algoritmo de líneas seguras ──
    lineas_corners = evaluar_lineas_seguras(corners_local + corners_visitante, UMBRALES_CORNERS)
    lineas_remates = evaluar_lineas_seguras(remates_arco_local + remates_arco_visitante, UMBRALES_REMATES_ARCO)

    # ── Mercado de Goles y BTTS (usando datos de matches ya cargados) ──
    goles_btts_local = analizar_mercado_goles_y_btts(partidos_local_casa)
    goles_btts_visitante = analizar_mercado_goles_y_btts(partidos_visitante_fuera)

    # Combinado: goles totales de ambos conjuntos históricos
    goles_totales_combinado = []
    for p in partidos_local_casa + partidos_visitante_fuera:
        score = p.get("score", {}) or {}
        hg = score.get("home")
        ag = score.get("away")
        if hg is not None and ag is not None:
            goles_totales_combinado.append(float(hg + ag))

    lineas_goles_combinado = evaluar_lineas_seguras(goles_totales_combinado, UMBRALES_GOLES_TOTALES)

    # ── Ensamblaje del nodo JSON ──
    nodo = {
        "match_id": match_id,
        "fecha": fecha,
        "hora_utc": match_obj.get("utc_date", ""),
        "liga": {
            "id": liga_id,
            "season_id": season_id
        },
        "equipo_local": {
            "id": home_id,
            "nombre": home_team.get("name")
        },
        "equipo_visitante": {
            "id": away_id,
            "nombre": away_team.get("name")
        },
        "historial_condicionado": {
            "local_casa_muestras": len(partidos_local_casa),
            "visitante_fuera_muestras": len(partidos_visitante_fuera)
        },
        "mercados": {
            "corners": {
                "promedio_local_casa": promedio_corners_local,
                "promedio_visitante_fuera": promedio_corners_visitante,
                "promedio_total": promedio_corners_total,
                "lineas": lineas_corners
            },
            "remates_al_arco": {
                "promedio_local_casa": promedio_remates_local,
                "promedio_visitante_fuera": promedio_remates_visitante,
                "promedio_total": promedio_remates_total,
                "lineas": lineas_remates
            },
            "goles_totales": {
                "promedio_total": round(np.mean(goles_totales_combinado), 2) if goles_totales_combinado else 0.0,
                "lineas": lineas_goles_combinado
            },
            "ambos_anotan": {
                "frecuencia_local_casa": goles_btts_local["btts_frecuencia"],
                "frecuencia_visitante_fuera": goles_btts_visitante["btts_frecuencia"],
                "seguro_local_casa": goles_btts_local["btts_seguro"],
                "seguro_visitante_fuera": goles_btts_visitante["btts_seguro"]
            },
            "tarjetas": {
                "promedio_local_casa": promedio_tarjetas_local,
                "promedio_visitante_fuera": promedio_tarjetas_visitante,
                "promedio_total": promedio_tarjetas_total
            }
        }
    }
    return nodo


def main():
    hoy_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logging.info(f"Iniciando análisis para fecha: {hoy_utc}")

    # Resolución dinámica de IDs de competiciones élite
    ids_ligas = resolver_ids_competiciones()
    if not ids_ligas:
        logging.error("No se pudieron resolver las competiciones élite. Abortando.")
        sys.exit(1)

    partidos_dia = obtener_partidos_del_dia(hoy_utc, ids_ligas)
    if not partidos_dia:
        logging.info("No se encontraron partidos élite para la fecha actual.")
        resultado_final = []
    else:
        logging.info(f"Se detectaron {len(partidos_dia)} partidos élite. Iniciando procesamiento...")
        resultado_final = []
        for idx, partido in enumerate(partidos_dia, start=1):
            logging.info(f"[{idx}/{len(partidos_dia)}] Analizando match...")
            nodo = procesar_partido(partido, ids_ligas)
            if nodo:
                resultado_final.append(nodo)
            # Pausa respetuosa entre partidos para no saturar la API
            time.sleep(1.0)

    # ── Exportación del JSON de salida ──
    salida = {
        "meta": {
            "fecha_generacion": datetime.now(timezone.utc).isoformat(),
            "total_partidos_analizados": len(resultado_final),
            "ligas_filtro": sorted(list(NOMBRES_LIGAS_ELITE)),
            "api_version": "v1.0.0",
            "proveedor": "thestatsapi.com"
        },
        "partidos": resultado_final
    }

    with open("top_partidos_hoy.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    logging.info("Archivo 'top_partidos_hoy.json' generado exitosamente.")
    logging.info(f"Partidos catalogados: {len(resultado_final)}")


if __name__ == "__main__":
    main()
