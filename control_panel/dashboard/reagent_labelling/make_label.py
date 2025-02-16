from PIL import Image, ImageDraw, ImageFont
import qrcode
import pandas as pd

# Define dimensions
width, height = 96, 230
MAPPING_TABLE_PATH = "../../mapping_table/General_mapping_sampler.csv"



# Load a slimmer font
font_path = "/System/Library/Fonts/Helvetica.ttc"


OUTPUT_DIR = "reagent_labelling/label_output/"
QR_URL_TEMPLATE = "http://192.168.1.172:8501/?loc="

# Load the mapping table
mapping_table = pd.read_csv(MAPPING_TABLE_PATH)

# skip the first row
for index, row in mapping_table.iterrows():

    if index == 0:
        continue
    # Create an image with white background
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_large = ImageFont.truetype(font_path, 40)  # Font size for well number
    font_small = ImageFont.truetype(font_path, 24)  # Font size for chemical name
    # Define text and QR code data
    well_number = row["Well_cor"]
    chemical_name = row["Reagent Name"]
    qr_data = QR_URL_TEMPLATE + well_number

    # Draw the well number
    draw.text((-2, 8), well_number, fill="black", font=font_large)

    # Wrap chemical name if necessary
    if draw.textbbox((0, 0), chemical_name, font=font_small)[2] > (width - 4):
        split_index = len(chemical_name) // 2
        chemical_name = chemical_name[:split_index] + "\n" + chemical_name[split_index:]

    # Check if wrapped text fits, adjust font size if necessary
    if draw.textbbox((0, 0), chemical_name, font=font_small)[2] > (width - 4):
        font_small = ImageFont.truetype(font_path, 16)

    # Check if wrapped text fits, adjust font size if necessary
    if draw.textbbox((0, 0), chemical_name, font=font_small)[3] > (height - 100):
        font_small = ImageFont.truetype(font_path, 10)
    if draw.textbbox((0, 0), chemical_name, font=font_small)[3] > (height - 100):
        split_index = len(chemical_name) // 2
        chemical_name = chemical_name[:split_index] + "\n" + chemical_name[split_index:]


    # Draw the chemical name
    draw.text((0, 60), chemical_name, fill="black", font=font_small)

    # Generate QR code
    qr = qrcode.QRCode(box_size=3)
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill="black", back_color="white")

    # Resize QR code
    qr_img = qr_img.resize((112, 112))

    # Paste QR code onto the label
    image.paste(qr_img, (-6, 126))

    # Save the image
    output_path = OUTPUT_DIR + well_number + ".png"
    image.save(output_path)
