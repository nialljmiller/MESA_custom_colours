#!/bin/bash
# Update all system packages (the --refresh flag forces metadata refresh)
sudo dnf upgrade --refresh -y

# Remove packages that were installed as dependencies but are no longer needed
sudo dnf autoremove -y

# Clean up the DNF cache to free up disk space
sudo dnf clean all

# (Optional) If you use Flatpak, update Flatpak applications
if command -v flatpak &> /dev/null; then
    flatpak update -y
fi

# (Optional) If you have an SSD, trim all mounted filesystems (adjust if needed)
sudo fstrim -av

# (Optional) Vacuum old systemd journal logs older than 2 weeks to save space
sudo journalctl --vacuum-time=2weeks
