#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Análisis Deportivo - API-Football v3
Autor: Backend Senior
Entorno: GitHub Actions / Cloud
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────

BASE_URL = "https://v3.football.api-sports.io"
API_KEY = os.environ.get("APIFOOTBALL_KEY", "")
if not API_KEY:
    logging.error("APIFOOTBALL_KEY no está configurada en las variables de entorno.")
    sys.exit(1)

HEADERS = {
    "x-apisports-key": API_KEY,
    "Accept": "application/json"
}

LIGAS_ELITE = {39, 140, 135, 78, 61, 2, 3, 13}
MAX_PARTIDOS_DIA = 10
MAX_PARTIDOS_HISTORICO = 10
MAX_FIXTURES_POR_EQUIPO = 25
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

def api_get(endpoint: str, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """
    Ejecuta una petición GET autenticada a API-Football v3 con reintentos
    exponenciales y manejo de rate-limit.
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
            if data.get("errors"):
                logging.error(f"Error de API en {url}: {data['errors']}")
            return data
        except requests.exceptions.RequestException as exc:
            logging.warning(f"Intento {intento}/{max_retries} fallido en {url}: {exc}")
            if intento == max_retries:
                logging.error(f"Agotados reintentos para {url}")
                raise
            time.sleep(1.5 ** intento)
    return {}


# ─────────────────────────────────────────────────────────────
# EXTRACTORES DE DATOS REALES
# ─────────────────────────────────────────────────────────────

def obtener_partidos_del_dia(fecha_str: str) -> List[Dict[str, Any]]:
    """
    Consulta /v3/fixtures?date={fecha_hoy} y filtra únicamente partidos
    pertenecientes a las ligas élite definidas.
    """
    data = api_get("/fixtures", {"date": fecha_str, "timezone": "UTC"})
    partidos = data.get("response", [])
    filtrados = []
    for p in partidos:
        league_id = p.get("league", {}).get("id")
        if league_id in LIGAS_ELITE:
            filtrados.append(p)
    return filtrados[:MAX_PARTIDOS_DIA]


def obtener_ultimos_partidos_equipo(team_id: int, es_local: bool) -> List[Dict[str, Any]]:
    """
    Consulta /v3/fixtures?team={team_id}&last=25&status=FT.
    Filtra exactamente los últimos 10 partidos finalizados ('FT') donde el
    equipo jugó en casa (es_local=True) o fuera (es_local=False).
    """
    data = api_get("/fixtures", {
        "team": team_id,
        "last": MAX_FIXTURES_POR_EQUIPO,
        "status": "FT"
    })
    partidos = data.get("response", [])
    filtrados = []
    for p in partidos:
        home_id = p.get("teams", {}).get("home", {}).get("id")
        away_id = p.get("teams", {}).get("away", {}).get("id")
        status_short = p.get("fixture", {}).get("status", {}).get("short", "")
        if status_short != "FT":
            continue
        if es_local and home_id == team_id:
            filtrados.append(p)
        elif not es_local and away_id == team_id:
            filtrados.append(p)
    return filtrados[:MAX_PARTIDOS_HISTORICO]


def obtener_estadisticas_fixture(fixture_id: int) -> Dict[str, Dict[str, Any]]:
    """
    Consulta /v3/fixtures/statistics?fixture={fixture_id}.
    Devuelve un diccionario mapeado por team_id con las métricas extraídas.
    """
    data = api_get("/fixtures/statistics", {"fixture": fixture_id})
    respuesta = data.get("response", [])
    stats_por_equipo = {}
    for entry in respuesta:
        team_id = entry.get("team", {}).get("id")
        stats_raw = entry.get("statistics", [])
        metricas = {
            "Corner Kicks": 0,
            "Shots on Goal": 0,
            "Total Shots": 0,
            "Yellow Cards": 0,
            "Red Cards": 0
        }
        for stat in stats_raw:
            tipo = stat.get("type", "")
            valor = stat.get("value")
            if tipo in metricas and valor is not None:
                try:
                    metricas[tipo] = int(valor)
                except (ValueError, TypeError):
                    metricas[tipo] = 0
        # Suma de tarjetas
        metricas["Total Cards"] = metricas["Yellow Cards"] + metricas["Red Cards"]
        stats_por_equipo[team_id] = metricas
    return stats_por_equipo


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
    Extrae goles totales y BTTS de los fixtures ya obtenidos (sin llamada adicional).
    """
    goles_totales = []
    btts_si = 0
    for p in partidos:
        home_goals = p.get("goals", {}).get("home") or 0
        away_goals = p.get("goals", {}).get("away") or 0
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

def procesar_partido(fixture_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Procesa un único partido del día: extrae historial condicionado,
    estadísticas de rendimiento, goles, BTTS y genera el nodo JSON.
    """
    fixture_id = fixture_obj.get("fixture", {}).get("id")
    fecha = fixture_obj.get("fixture", {}).get("date", "")[:10]
    liga = fixture_obj.get("league", {})
    teams = fixture_obj.get("teams", {})
    home_team = teams.get("home", {})
    away_team = teams.get("away", {})
    home_id = home_team.get("id")
    away_id = away_team.get("id")

    logging.info(f"Procesando partido {fixture_id}: {home_team.get('name')} vs {away_team.get('name')}")

    # ── Extracción de historial condicionado ──
    partidos_local_casa = obtener_ultimos_partidos_equipo(home_id, es_local=True)
    partidos_visitante_fuera = obtener_ultimos_partidos_equipo(away_id, es_local=False)

    if len(partidos_local_casa) < 5 or len(partidos_visitante_fuera) < 5:
        logging.warning(f"Datos insuficientes para {fixture_id}. Se requieren mínimo 5 partidos históricos por bando.")
        return None

    # ── Recolección de estadísticas de rendimiento ──
    corners_local = []
    remates_arco_local = []
    tarjetas_local = []

    for p in partidos_local_casa:
        fid = p.get("fixture", {}).get("id")
        stats = obtener_estadisticas_fixture(fid)
        home_stats = stats.get(home_id, {})
        corners_local.append(float(home_stats.get("Corner Kicks", 0)))
        remates_arco_local.append(float(home_stats.get("Shots on Goal", 0)))
        tarjetas_local.append(float(home_stats.get("Total Cards", 0)))

    corners_visitante = []
    remates_arco_visitante = []
    tarjetas_visitante = []

    for p in partidos_visitante_fuera:
        fid = p.get("fixture", {}).get("id")
        stats = obtener_estadisticas_fixture(fid)
        away_stats = stats.get(away_id, {})
        corners_visitante.append(float(away_stats.get("Corner Kicks", 0)))
        remates_arco_visitante.append(float(away_stats.get("Shots on Goal", 0)))
        tarjetas_visitante.append(float(away_stats.get("Total Cards", 0)))

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

    # ── Mercado de Goles y BTTS (usando datos de fixtures ya cargados) ──
    goles_btts_local = analizar_mercado_goles_y_btts(partidos_local_casa)
    goles_btts_visitante = analizar_mercado_goles_y_btts(partidos_visitante_fuera)

    # Combinado: goles totales de ambos conjuntos históricos
    goles_totales_combinado = []
    for p in partidos_local_casa + partidos_visitante_fuera:
        hg = p.get("goals", {}).get("home") or 0
        ag = p.get("goals", {}).get("away") or 0
        goles_totales_combinado.append(float(hg + ag))

    lineas_goles_combinado = evaluar_lineas_seguras(goles_totales_combinado, UMBRALES_GOLES_TOTALES)

    # ── Ensamblaje del nodo JSON ──
    nodo = {
        "fixture_id": fixture_id,
        "fecha": fecha,
        "hora_utc": fixture_obj.get("fixture", {}).get("date", ""),
        "liga": {
            "id": liga.get("id"),
            "nombre": liga.get("name"),
            "pais": liga.get("country"),
            "temporada": liga.get("season")
        },
        "equipo_local": {
            "id": home_id,
            "nombre": home_team.get("name"),
            "logo": home_team.get("logo")
        },
        "equipo_visitante": {
            "id": away_id,
            "nombre": away_team.get("name"),
            "logo": away_team.get("logo")
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

    partidos_dia = obtener_partidos_del_dia(hoy_utc)
    if not partidos_dia:
        logging.info("No se encontraron partidos élite para la fecha actual.")
        resultado_final = []
    else:
        logging.info(f"Se detectaron {len(partidos_dia)} partidos élite. Iniciando procesamiento...")
        resultado_final = []
        for idx, partido in enumerate(partidos_dia, start=1):
            logging.info(f"[{idx}/{len(partidos_dia)}] Analizando fixture...")
            nodo = procesar_partido(partido)
            if nodo:
                resultado_final.append(nodo)
            # Pausa respetuosa entre partidos para no saturar la API
            time.sleep(1.2)

    # ── Exportación del JSON de salida ──
    salida = {
        "meta": {
            "fecha_generacion": datetime.now(timezone.utc).isoformat(),
            "total_partidos_analizados": len(resultado_final),
            "ligas_filtro": sorted(list(LIGAS_ELITE)),
            "api_version": "v3",
            "proveedor": "api-football.com"
        },
        "partidos": resultado_final
    }

    with open("top_partidos_hoy.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    logging.info("Archivo 'top_partidos_hoy.json' generado exitosamente.")
    logging.info(f"Partidos catalogados: {len(resultado_final)}")


if __name__ == "__main__":
    main()
