import json
from typing import Dict, Optional
from langchain.tools import StructuredTool

# Wczytanie danych o wycieczkach
with open('data/trips_data.json', 'r', encoding='utf-8') as f:
    trips_data = json.load(f)

def fetch_trip_details(
    country: Optional[str] = None,
    city: Optional[str] = None,
    start_date: Optional[str] = None,
    trip_id: Optional[int] = None
) -> Dict:
    """
    Fetch details about available trips based on search criteria.
    
    Args:
        country: Country to search for (e.g., "Italy")
        city: City to search for (e.g., "Rome")
        start_date: Trip start date in YYYY-MM-DD format
        trip_id: Numeric ID of the trip (index in the list)
    
    Returns:
        Dictionary with trip details or error message
    """
    try:
        if trip_id is not None:
            if 0 <= trip_id < len(trips_data):
                return trips_data[trip_id]
            return {"error": f"No trip found with ID {trip_id}"}
        
        if country and not any([city, start_date, trip_id]):
            return {
                "results": [trip for trip in trips_data 
                        if trip["Country"].lower() == country.lower()]
            }
        
        results = []
        for trip in trips_data:
            match = True
            
            if country and trip['Country'].lower() != country.lower():
                match = False
                
            if city and trip['City'].lower() != city.lower():
                match = False
                
            if start_date and trip['Start date'] != start_date:
                match = False
                
            if match:
                results.append(trip)
        
        if not results:
            return {"error": "No trips found matching the criteria"}
        
        return {"results": results}
    
    except Exception as e:
        return {"error": f"Error searching trips: {str(e)}"}

# Eksport narzędzia dla agenta
fetch_trip_details_tool = StructuredTool.from_function(
    func=fetch_trip_details,
    name="fetch_trip_details",
    description="""
    Use this tool to search for available trips based on criteria.
    Input arguments (at least one required):
    - country: Country name (e.g., "France")
    - city: City name (e.g., "Paris")
    - start_date: Trip start date in YYYY-MM-DD format
    - trip_id: Numeric ID of the trip
    Returns trip details or list of matching trips.
    """
)