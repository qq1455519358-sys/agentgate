"""
Airline booking agent — LangGraph-based.

This is a reference implementation. To run it, you need:
    pip install agentgate[langgraph] langchain-openai
    export OPENAI_API_KEY=...

For mock-mode testing (no API keys), see test_booking.py.
"""

from __future__ import annotations

# Tool implementations (these work without LLM)
BOOKINGS = {
    "BK001": {"status": "confirmed", "flight": "NH101", "passenger": "John"},
    "BK002": {"status": "pending", "flight": "UA202", "passenger": "Jane"},
}

FLIGHTS = [
    {"flight": "NH101", "from": "SFO", "to": "NRT", "price": 850},
    {"flight": "UA202", "from": "SFO", "to": "NRT", "price": 920},
    {"flight": "JL003", "from": "SFO", "to": "NRT", "price": 780},
]


def search_flights(origin: str, destination: str) -> str:
    matches = [f for f in FLIGHTS if f["from"] == origin and f["to"] == destination]
    if matches:
        return f"Found {len(matches)} flights: " + ", ".join(
            f"{f['flight']} (${f['price']})" for f in matches
        )
    return f"No flights found from {origin} to {destination}"


def check_booking(booking_id: str) -> str:
    if booking_id in BOOKINGS:
        b = BOOKINGS[booking_id]
        return f"Booking {booking_id}: {b}"
    return f"Booking {booking_id} not found"


def cancel_booking(booking_id: str) -> str:
    if booking_id in BOOKINGS:
        return f"Booking {booking_id} has been cancelled. Refund will be processed."
    return f"Booking {booking_id} not found"


def book_flight(flight_id: str, passenger: str) -> str:
    return f"Booked {flight_id} for {passenger}. Confirmation: BK{hash(flight_id) % 1000:03d}"
