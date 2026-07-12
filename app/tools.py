"""
Dynamic Tools Layer
---------------------
Real-time data sources the chatbot can call when a question needs
current information rather than something from the static knowledge
base. Uses Open-Meteo (free, no API key) as a concrete example of a
live API call — swap in or add other tools (stock prices, ticketing
systems, internal APIs, etc.) the same way: one function per tool, each
returning a short string suitable for dropping straight into the prompt.
"""

from datetime import datetime, timezone

import requests


def get_current_datetime(_args: dict) -> str:
    now = datetime.now(timezone.utc)
    return f"Current UTC date/time: {now.strftime('%Y-%m-%d %H:%M')} UTC"


def get_weather(args: dict) -> str:
    """Looks up current weather for a city using Open-Meteo's free geocoding + forecast APIs."""
    city = args.get("city", "").strip()
    if not city:
        return "No city was specified for the weather lookup."

    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=8,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results")
        if not results:
            return f"Could not find a location named '{city}'."

        lat, lon = results[0]["latitude"], results[0]["longitude"]
        resolved_name = results[0].get("name", city)

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
            timeout=8,
        )
        weather_resp.raise_for_status()
        cw = weather_resp.json().get("current_weather", {})
        if not cw:
            return f"Weather data unavailable for {resolved_name}."

        return (
            f"Current weather in {resolved_name}: {cw.get('temperature')}°C, "
            f"wind {cw.get('windspeed')} km/h."
        )
    except requests.RequestException as e:
        return f"Weather lookup failed: {e}"


# Registry of available tools: name -> (description for the router, function)
TOOLS = {
    "get_weather": {
        "description": "Look up current weather for a named city. Args: {city: string}",
        "fn": get_weather,
    },
    "get_current_datetime": {
        "description": "Get the current UTC date and time. Args: {}",
        "fn": get_current_datetime,
    },
}


def run_tool(name: str, args: dict) -> str:
    tool = TOOLS.get(name)
    if not tool:
        return f"Unknown tool: {name}"
    return tool["fn"](args)


def tools_description() -> str:
    return "\n".join(f"- {name}: {info['description']}" for name, info in TOOLS.items())
