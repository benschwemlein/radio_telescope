\# Extract
tar -xzf radio_telescope_FIXED.tar.gz

# Move directories
mv current/database .
mv current/radio_telescope .
mv current/ui/scan_dialog.py ui/

# Replace files with FIXED versions
mv main_window_fixed.py ui/main_window.py
mv star_chart_view_fixed.py ui/star_chart_view.py

# Clean up
rm -rf current/

# Run
python3 app.py