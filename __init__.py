"""Expose all Livepeer nodes for ComfyUI."""

import logging
import pathlib
import importlib
import traceback
import os
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Store all mappings centrally
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Determine the directory containing this __init__.py file
EXTENSION_ROOT = os.path.dirname(os.path.abspath(__file__))
NODE_WRAPPER_DIR = os.path.join(EXTENSION_ROOT, "node_wrappers")

# Add the extension root to sys.path FOR the imports
# Prevent interference with other packages
sys.path.insert(0, EXTENSION_ROOT)

def load_nodes(module_path_rel: str, module_name: str):
    """Loads nodes from a given module, handling potential errors."""
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    try:
        module = importlib.import_module(module_path_rel, package=__name__)
        log.info(f"Imported module: {module_name}")
        
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            log.debug(f"Updated NODE_CLASS_MAPPINGS from {module_name}")
            
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                log.debug(f"Updated NODE_DISPLAY_NAME_MAPPINGS from {module_name}")
            else:
                 # Use class name if display name mapping is missing
                 for name in module.NODE_CLASS_MAPPINGS:
                      if name not in NODE_DISPLAY_NAME_MAPPINGS:
                           NODE_DISPLAY_NAME_MAPPINGS[name] = name
                           log.warning(f"Missing display name for '{name}' in {module_name}, using class name.")
                           
        else:
             log.debug(f"No NODE_CLASS_MAPPINGS found in module: {module_name}")
             
    except ImportError as e:
        log.error(f"Failed to import module {module_name}: {e}\n{traceback.format_exc()}")
    except Exception as e:
        log.error(f"Error loading nodes from module {module_name}: {e}\n{traceback.format_exc()}")

# Recursively find and load nodes from subdirectories
for root, dirs, files in os.walk(NODE_WRAPPER_DIR):
    # Skip __pycache__ directories
    if "__pycache__" in dirs:
        dirs.remove("__pycache__")
        
    for filename in files:
        if filename.endswith(".py") and not filename.startswith("__") and not filename.startswith("base_"):
            module_name = filename[:-3] # Remove .py extension
            
            # Construct the relative module path (e.g., '.node_wrappers.image.livepeer_t2i')
            relative_root = os.path.relpath(root, EXTENSION_ROOT)
            # Replace os separators with dots
            module_path_rel_parts = relative_root.split(os.sep)
            module_path_rel = "." + ".".join(module_path_rel_parts) + "." + module_name
            
            log.debug(f"Attempting to load nodes from: {module_path_rel}")
            load_nodes(module_path_rel, module_name)

# For backward compatibility, also try to load the batch nodes
try:
    import importlib.util
    batch_iterator_path = os.path.join(EXTENSION_ROOT, "batch_iterator.py")
    batch_info_path = os.path.join(EXTENSION_ROOT, "batch_info.py")
    
    # Check if these files still exist (they might have been moved to node_wrappers)
    if os.path.exists(batch_iterator_path):
        module_name = "batch_iterator"
        load_nodes(f".{module_name}", module_name)
    
    if os.path.exists(batch_info_path):
        module_name = "batch_info"
        load_nodes(f".{module_name}", module_name)
except Exception as e:
    log.warning(f"Error loading batch utilities: {e}")

# Clean up sys.path by removing the extension root
try:
    sys.path.remove(EXTENSION_ROOT)
except ValueError:
    pass # In case it was already removed or not present

log.info("--- Livepeer Nodes Load Complete ---")
log.info(f"Loaded {len(NODE_CLASS_MAPPINGS)} node classes.")

# --- Web UI Integration (Optional) ---
WEB_DIRECTORY = "./js" # Relative path to the directory containing your JS files

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 