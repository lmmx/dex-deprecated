# Processing index page images for Strang (2019) Linear Algebra and Learning from Data

- [x] Image files were taken on my phone camera, stored in `img/`
- [x] Bash script renamed them to their page numbers (see `src/rename_images.sh`)
  - Index of authors: `420.jpg` - `422.jpg`
  - Index of topics:  `423.jpg` - `431.jpg`
  - Index of symbols: `432.jpg`
- [ ] Python script `src/process_scans.py` crops, sharpens, and increases
  the contrast of the 'scan' images, then assembles them into a single PDF
  (see `doc/index_auth.pdf`, `doc/index_topics.pdf`, `doc/index_symbols.pdf`)
- [ ] Tesseract was run on the [author and topic] index PDFs to add an OCR annotation
  text layer
