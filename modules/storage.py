import json
import streamlit as st
import datetime
from typing import Dict, Optional, Any

# Try to import Replit Object Storage
try:
    from replit.object_storage import Client
    object_storage_available = True
except ImportError:
    object_storage_available = False

def check_storage_connection() -> bool:
    """
    Check if Replit Object Storage is available and connected
    
    Returns:
        Boolean indicating if storage is connected
    """
    if not object_storage_available:
        return False
    
    try:
        # Try to initialize and connect to Replit Object Storage
        client = Client()
        # Test connection by listing contents
        client.list()
        return True
    except Exception as e:
        st.error(f"Failed to connect to Replit Object Storage: {str(e)}")
        return False

def load_best_params(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Load best parameters for a symbol from Replit Object Storage
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USD')
        
    Returns:
        Dict of parameters or None if not found
    """
    best_params_file = "best_params.json"
    
    try:
        # Try to get parameters from Replit Object Storage first
        if object_storage_available:
            connected = check_storage_connection()
            if not connected:
                st.warning("Replit Object Storage not available. Trying local file...")
                
            # Try to get parameters from Object Storage
            try:
                client = Client()
                json_content = client.download_as_text("best_params.json")
                best_params_data = json.loads(json_content)
                
                if symbol in best_params_data:
                    params = best_params_data[symbol]['best_params']
                    return params
                else:
                    st.warning(f"No optimized parameters found for {symbol} in storage. Using defaults.")
                    return None
            except Exception as e:
                st.warning(f"Error accessing Replit Object Storage: {str(e)}. Falling back to local file.")
                
                # Fallback to local file if needed
                try:
                    with open(best_params_file, "r") as f:
                        best_params_data = json.load(f)
                        
                        if symbol in best_params_data:
                            params = best_params_data[symbol]['best_params']
                            return params
                        else:
                            st.warning(f"No optimized parameters found for {symbol}. Using defaults.")
                            return None
                except FileNotFoundError:
                    st.warning("Best parameters file not found. Using defaults.")
                    return None
    except Exception as e:
        st.warning(f"Error loading parameters: {str(e)}. Using defaults.")
        return None

def save_best_params(symbol: str, params: Dict[str, Any]) -> bool:
    """
    Save best parameters for a symbol to Replit Object Storage
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USD')
        params: Parameters to save
        
    Returns:
        Boolean indicating success
    """
    best_params_file = "best_params.json"
    
    try:
        # Try to load existing parameters
        existing_params = {}
        
        if object_storage_available:
            # Try to get existing parameters from Object Storage
            try:
                client = Client()
                try:
                    json_content = client.download_as_text("best_params.json")
                    existing_params = json.loads(json_content)
                except:
                    # File doesn't exist yet
                    pass
                
                # Update with new parameters
                existing_params[symbol] = {
                    'best_params': params,
                    'timestamp': str(datetime.datetime.now())
                }
                
                # Save back to Object Storage
                client.upload_text("best_params.json", json.dumps(existing_params, indent=2))
                return True
            except Exception as e:
                st.warning(f"Error saving to Replit Object Storage: {str(e)}. Saving to local file...")
        
        # Fallback to local file
        try:
            try:
                with open(best_params_file, "r") as f:
                    existing_params = json.load(f)
            except FileNotFoundError:
                pass
            
            # Update with new parameters
            existing_params[symbol] = {
                'best_params': params,
                'timestamp': str(datetime.datetime.now())
            }
            
            # Save to local file
            with open(best_params_file, "w") as f:
                json.dump(existing_params, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Error saving parameters to local file: {str(e)}")
            return False
    except Exception as e:
        st.error(f"Error saving parameters: {str(e)}")
        return False