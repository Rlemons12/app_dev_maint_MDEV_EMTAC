import subprocess
import pkg_resources

def is_installed(package_name):
    """Check if a Python package is installed."""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_package(package_name):
    """Install a Python package using pip."""
    subprocess.check_call(["pip", "install", package_name])

# List of third-party libraries you provided
third_party_libraries = [
    "openai",
    "fuzzywuzzy",
    "werkzeug",
    "flask",
    "pandas"
]

for library in third_party_libraries:
    if not is_installed(library):
        print(f"{library} is not installed. Installing now...")
        install_package(library)
    else:
        print(f"{library} is already installed.")

print("Checking and installation process completed!")
