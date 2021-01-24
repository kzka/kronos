FILE_IDS=(
    1b46_5-g8RGgLBsKFzCDxoP3xsbomQ36r
)
echo "Downloading..."
for id in "${FILE_IDS[@]}"; do
    gdown "https://drive.google.com/uc?export=download&id=${id}"
done
echo "Unzipping..."
unzip Penn_Action.zip
cd ..
mkdir -p data
mv scripts/Penn_Action data/
echo "Done."
