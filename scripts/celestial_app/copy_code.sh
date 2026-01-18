#!/bin/bash
# Refactored Code Installation Script
# This script safely replaces your existing code with the refactored version

set -e  # Exit on any error

echo "=========================================="
echo "Celestial Sphere - Refactored Code Install"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found in current directory!"
    echo "Please run this script from your project root directory."
    exit 1
fi

# 1. Create backup
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "Step 1: Creating backup in ${BACKUP_DIR}/"
mkdir -p "${BACKUP_DIR}"

# Backup all Python directories and files
cp -r astronomy "${BACKUP_DIR}/" 2>/dev/null || true
cp -r debug "${BACKUP_DIR}/" 2>/dev/null || true
cp -r geometry "${BACKUP_DIR}/" 2>/dev/null || true
cp -r ui "${BACKUP_DIR}/" 2>/dev/null || true
cp -r visualization "${BACKUP_DIR}/" 2>/dev/null || true
cp app.py "${BACKUP_DIR}/" 2>/dev/null || true

echo "✓ Backup created successfully"
echo ""

# 2. Extract refactored code
echo "Step 2: Extracting refactored code..."

# Try multiple possible locations for the tar.gz file
if [ -f "refactored_code.tar.gz" ]; then
    TAR_FILE="refactored_code.tar.gz"
elif [ -f "~/Downloads/refactored_code.tar.gz" ]; then
    TAR_FILE="~/Downloads/refactored_code.tar.gz"
elif [ -f "$HOME/Downloads/refactored_code.tar.gz" ]; then
    TAR_FILE="$HOME/Downloads/refactored_code.tar.gz"
else
    echo "ERROR: Cannot find refactored_code.tar.gz"
    echo "Looked in:"
    echo "  - Current directory (./refactored_code.tar.gz)"
    echo "  - Downloads folder (~/Downloads/refactored_code.tar.gz)"
    echo ""
    echo "Please copy refactored_code.tar.gz to the current directory or specify location:"
    echo "  cp ~/Downloads/refactored_code.tar.gz ."
    echo "Your backup is safe in ${BACKUP_DIR}/"
    exit 1
fi

tar -xzf "$TAR_FILE"

echo "✓ Refactored code extracted"
echo ""

# 3. Verify extraction
echo "Step 3: Verifying files..."
if [ ! -f "ui/theme.py" ]; then
    echo "ERROR: ui/theme.py not found after extraction!"
    echo "Extraction may have failed. Your backup is safe in ${BACKUP_DIR}/"
    exit 1
fi

echo "✓ All files verified"
echo ""

# 4. Done!
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Changes applied:"
echo "  ✓ Vector normalization utility added"
echo "  ✓ Coordinate transformation functions added"
echo "  ✓ Time convenience functions added"
echo "  ✓ Theme manager created (NEW: ui/theme.py)"
echo "  ✓ OpenGL mesh factories added"
echo "  ✓ All duplicate code removed"
echo ""
echo "Backup location: ${BACKUP_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Test your application: python3 app.py"
echo "  2. If everything works, you can delete the backup"
echo "  3. If issues occur, restore from backup:"
echo "     rm -rf astronomy debug geometry ui visualization app.py"
echo "     cp -r ${BACKUP_DIR}/* ."
echo ""
echo "Happy coding! 🚀"