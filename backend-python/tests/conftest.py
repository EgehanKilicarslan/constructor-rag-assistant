import os
import sys

# Add project root directory and pb folder to sys.path
# This allows tests to find 'app' and 'pb' modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../pb")))
