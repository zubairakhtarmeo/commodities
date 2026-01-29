"""
Event & Market Intelligence Collector for Commodities
Collects news, events, geopolitics, weather affecting commodity prices
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import feedparser  # RSS feed parser
from bs4 import BeautifulSoup  # For web scraping

# API Configuration - FREE APIs only
WEATHER_API = "https://wttr.in/{}?format=j1"  # Free weather API, no key needed
WEATHER_BACKUP_API = "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&current=temperature_2m,relative_humidity_2m,weather_code"  # Free weather API

# Location coordinates for weather (latitude, longitude)
LOCATIONS = {
    "Multan": (30.1575, 71.5249),
    "Karachi": (24.8607, 67.0011),
    "Hyderabad": (25.3792, 68.3683)
}

NEWS_RSS_FEEDS = {
    "reuters_commodities": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
    "bloomberg_markets": "https://www.bloomberg.com/feed/podcast/markets-daily.xml",
}

# Commodity keywords for tracking
COMMODITY_KEYWORDS = {
    "cotton": ["cotton", "textile", "fabric", "yarn", "cotton harvest", "cotton production"],
    "polyester": ["polyester", "PET", "polymer", "synthetic fiber", "terephthalate"],
    "oil": ["crude oil", "brent", "WTI", "petroleum", "OPEC", "oil price"],
    "gas": ["natural gas", "LNG", "gas price", "gas supply"],
    "energy": ["energy crisis", "electricity", "power tariff", "NEPRA"],
    "pakistan": ["Pakistan economy", "USD PKR", "rupee", "State Bank Pakistan", "imports Pakistan"]
}

# Event categories
EVENT_IMPACT = {
    "geopolitics": ["war", "sanctions", "conflict", "trade war", "embargo", "tension"],
    "weather": ["drought", "flood", "hurricane", "cyclone", "monsoon", "heat wave"],
    "policy": ["tariff", "quota", "ban", "regulation", "subsidy", "tax"],
    "supply": ["shortage", "disruption", "strike", "port congestion", "shipping"],
    "demand": ["demand surge", "consumption", "slowdown", "recession"]
}

DATA_DIR = Path("data/events")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_commodity_news(commodity: str, days_back: int = 7) -> List[Dict]:
    """Fetch news for specific commodity using free sources."""
    events = []
    
    # Method 1: Try Google News RSS (free, no API key)
    try:
        keywords = "+".join(COMMODITY_KEYWORDS.get(commodity, [commodity])[:3])
        # URL encode properly to avoid spaces
        keywords = keywords.replace(" ", "+")
        google_news_url = f"https://news.google.com/rss/search?q={keywords}+commodity+price&hl=en-PK&gl=PK&ceid=PK:en"
        
        feed = feedparser.parse(google_news_url)
        for entry in feed.entries[:5]:
            pub_date = entry.get('published', datetime.now().isoformat())
            events.append({
                "timestamp": pub_date,
                "commodity": commodity,
                "title": entry.get('title', 'No title'),
                "description": entry.get('summary', '')[:200],
                "source": entry.get('source', {}).get('title', 'Google News'),
                "url": entry.get('link', ''),
                "category": classify_event(entry.get('title', '') + " " + entry.get('summary', '')),
                "collected_at": datetime.now().isoformat()
            })
    except Exception as e:
        print(f"Google News RSS error for {commodity}: {e}")
    
    # Method 2: Try Bing News RSS (backup)
    if not events:
        try:
            keywords = "+".join(COMMODITY_KEYWORDS.get(commodity, [commodity])[:2])
            keywords = keywords.replace(" ", "+")
            bing_url = f"https://www.bing.com/news/search?q={keywords}&format=rss"
            
            feed = feedparser.parse(bing_url)
            for entry in feed.entries[:3]:
                events.append({
                    "timestamp": entry.get('published', datetime.now().isoformat()),
                    "commodity": commodity,
                    "title": entry.get('title', 'No title'),
                    "description": entry.get('description', '')[:200],
                    "source": "Bing News",
                    "url": entry.get('link', ''),
                    "category": classify_event(entry.get('title', '')),
                    "collected_at": datetime.now().isoformat()
                })
        except Exception as e:
            print(f"Bing News error for {commodity}: {e}")
    
    return events if events else get_fallback_events(commodity)


def classify_event(text: str) -> str:
    """Classify event into impact category."""
    text_lower = text.lower()
    for category, keywords in EVENT_IMPACT.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "general"


def get_fallback_events(commodity: str) -> List[Dict]:
    """Get synthetic/example events when API not available."""
    # This would scrape RSS feeds or use alternative sources
    # For now, return structure for testing
    return [{
        "timestamp": datetime.now().isoformat(),
        "commodity": commodity,
        "title": f"Monitor {commodity.title()} market conditions",
        "description": "API key required for live news. Add NewsAPI key to enable.",
        "source": "System",
        "url": "https://newsapi.org",
        "category": "general",
        "collected_at": datetime.now().isoformat()
    }]


def fetch_weather_alerts() -> List[Dict]:
    """Fetch weather alerts affecting commodity production using FREE API."""
    alerts = []
    
    # Cotton producing regions in Pakistan
    cotton_regions = [
        {"city": "Multan", "region": "Punjab, Pakistan", "crop": "cotton"},
        {"city": "Karachi", "region": "Sindh, Pakistan", "crop": "cotton"},
        {"city": "Hyderabad", "region": "Sindh, Pakistan", "crop": "cotton"}
    ]
    
    for region_info in cotton_regions:
        try:
            # Use wttr.in free weather API (no key required)
            response = requests.get(WEATHER_API.format(region_info['city']), timeout=5)
            if response.status_code == 200:
                weather_data = response.json()
                
                current = weather_data.get('current_condition', [{}])[0]
                temp_c = current.get('temp_C', 'N/A')
                humidity = current.get('humidity', 'N/A')
                weather_desc = current.get('weatherDesc', [{}])[0].get('value', 'Unknown')
                
                # Check for adverse conditions
                severity = "info"
                alert_msg = f"Current: {temp_c}°C, {humidity}% humidity, {weather_desc}"
                
                if int(humidity) > 80:
                    severity = "warning"
                    alert_msg += " ⚠️ High humidity may affect cotton quality"
                elif int(temp_c) > 40:
                    severity = "warning"
                    alert_msg += " ⚠️ Extreme heat may stress crops"
                
                alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "weather",
                    "region": region_info["region"],
                    "crop": region_info["crop"],
                    "alert": alert_msg,
                    "severity": severity,
                    "temperature": temp_c,
                    "humidity": humidity,
                    "condition": weather_desc,
                    "collected_at": datetime.now().isoformat()
                })
                continue
        except:
            pass
        
        # Backup: Try open-meteo (free, no key required)
        try:
            lat, lon = LOCATIONS.get(region_info['city'], (0, 0))
            response = requests.get(WEATHER_BACKUP_API.format(lat, lon), timeout=5)
            if response.status_code == 200:
                data = response.json()
                current = data.get('current', {})
                temp = current.get('temperature_2m', 'N/A')
                humidity = current.get('relative_humidity_2m', 'N/A')
                
                alert_msg = f"Current: {temp}°C, {humidity}% humidity"
                severity = "info"
                
                alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "weather",
                    "region": region_info["region"],
                    "crop": region_info["crop"],
                    "alert": alert_msg,
                    "severity": severity,
                    "temperature": temp,
                    "humidity": humidity,
                    "collected_at": datetime.now().isoformat()
                })
                continue
        except Exception as e:
            print(f"Backup weather API error for {region_info['city']}: {e}")
        
        # Final fallback
        alerts.append({
            "timestamp": datetime.now().isoformat(),
            "type": "weather",
            "region": region_info["region"],
            "crop": region_info["crop"],
            "alert": "Weather monitoring active",
            "severity": "info",
            "collected_at": datetime.now().isoformat()
        })
    
    return alerts


def fetch_geopolitical_events() -> List[Dict]:
    """Track geopolitical events affecting trade."""
    events = []
    
    # Would integrate with GDELT, ACLED for conflict data
    # Placeholder for key monitoring areas
    watch_areas = [
        {"region": "Middle East", "impact": ["oil", "gas"], "status": "monitor"},
        {"region": "Red Sea", "impact": ["oil", "shipping"], "status": "monitor"},
        {"region": "Ukraine", "impact": ["oil", "gas", "wheat"], "status": "monitor"}
    ]
    
    for area in watch_areas:
        events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "geopolitical",
            "region": area["region"],
            "affected_commodities": area["impact"],
            "status": area["status"],
            "collected_at": datetime.now().isoformat()
        })
    
    return events


def collect_all_events() -> Dict:
    """Main collection function."""
    print("🔄 Starting event collection...")
    
    all_data = {
        "collection_time": datetime.now().isoformat(),
        "news": {},
        "weather": [],
        "geopolitical": []
    }
    
    # Collect news for each commodity
    for commodity in ["cotton", "polyester", "oil", "gas", "pakistan"]:
        print(f"  📰 Collecting {commodity} news...")
        all_data["news"][commodity] = fetch_commodity_news(commodity)
    
    # Collect weather alerts
    print("  🌦️ Collecting weather alerts...")
    all_data["weather"] = fetch_weather_alerts()
    
    # Collect geopolitical events
    print("  🌍 Collecting geopolitical events...")
    all_data["geopolitical"] = fetch_geopolitical_events()
    
    # Save to file
    output_file = DATA_DIR / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Events saved to: {output_file}")
    
    # Also save latest as current
    latest_file = DATA_DIR / "events_latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    return all_data


def get_latest_events() -> Dict:
    """Load latest collected events."""
    latest_file = DATA_DIR / "events_latest.json"
    if latest_file.exists():
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"news": {}, "weather": [], "geopolitical": []}


def get_critical_alerts(events: Dict) -> List[Dict]:
    """Extract critical alerts requiring immediate attention."""
    alerts = []
    
    # Check news for high-impact keywords
    high_impact_keywords = ["surge", "crisis", "shortage", "ban", "war", "disruption"]
    
    for commodity, news_items in events.get("news", {}).items():
        for item in news_items:
            text = (item.get("title", "") + " " + item.get("description", "")).lower()
            if any(keyword in text for keyword in high_impact_keywords):
                alerts.append({
                    "severity": "high",
                    "commodity": commodity,
                    "title": item.get("title"),
                    "category": item.get("category"),
                    "timestamp": item.get("timestamp")
                })
    
    return alerts


if __name__ == "__main__":
    print("=" * 60)
    print("Commodity Event & Intelligence Collector")
    print("=" * 60)
    
    data = collect_all_events()
    
    # Show summary
    print("\n📊 Collection Summary:")
    for commodity, items in data["news"].items():
        print(f"  {commodity}: {len(items)} news items")
    print(f"  Weather alerts: {len(data['weather'])}")
    print(f"  Geopolitical events: {len(data['geopolitical'])}")
    
    # Show critical alerts
    alerts = get_critical_alerts(data)
    if alerts:
        print(f"\n⚠️  {len(alerts)} Critical Alerts Found!")
        for alert in alerts[:3]:
            print(f"    • [{alert['commodity']}] {alert['title']}")
    
    print("\n✅ Collection complete!")
