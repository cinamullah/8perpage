import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# --- Logic ---

def improve_image(image_bytes, max_kb=500):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    kernel = np.ones((15, 15), np.uint8)
    background = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    background = cv2.GaussianBlur(background, (15, 15), 0)
    res = cv2.divide(img, background, scale=255)
    
    magic_color = cv2.convertScaleAbs(res, alpha=1.1, beta=-10)
    hsv = cv2.cvtColor(magic_color, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] *= 1.4
    hsv[:, :, 2] *= 1.1
    hsv = np.clip(hsv, 0, 255).astype("uint8")
    final_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    final_output = cv2.filter2D(final_output, -1, sharpen_kernel)
    
    _, encimg = cv2.imencode('.jpg', final_output, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return encimg.tobytes()

def generate_cnic_pdf(front_bytes, back_bytes):
    w, h = A4
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    IMG_W, IMG_H = 9.5 * cm, 6.5 * cm
    MARGIN, GAP = 0.5 * cm, 0.4 * cm

    def draw_page(image_bytes):
        img_reader = ImageReader(BytesIO(image_bytes))
        for i in range(8):
            col, row = i % 2, i // 2
            x = MARGIN + col * (IMG_W + GAP)
            y = h - MARGIN - IMG_H - row * (IMG_H + GAP)
            c.drawImage(img_reader, x, y, IMG_W, IMG_H)

    draw_page(front_bytes)
    c.showPage()
    draw_page(back_bytes)
    c.showPage()
    c.save()
    return pdf_buffer.getvalue()

# --- Minimalist UI ---

def main():
    st.set_page_config(page_title="8perPage", page_icon="🆔", layout="centered")
    
    # Hide standard Streamlit header/footer
    st.markdown("""<style>.stDeployButton, footer {visibility: hidden;}</style>""", unsafe_allow_html=True)

    st.title("🆔 8perPage")
    
    # Side-by-side inputs without previews
    col1, col2 = st.columns(2)
    with col1:
        front_file = st.file_uploader("Front", type=["jpg", "png"], label_visibility="visible")
    with col2:
        back_file = st.file_uploader("Back", type=["jpg", "png"], label_visibility="visible")

    improve = st.checkbox("Apply Magic Enhancement", value=True)

    if st.button("Generate PDF", type="primary", use_container_width=True):
        if not front_file or not back_file:
            st.error("Please upload both sides.")
        else:
            with st.spinner("Processing..."):
                f_bytes = front_file.read()
                b_bytes = back_file.read()

                if improve:
                    f_bytes = improve_image(f_bytes) or f_bytes
                    b_bytes = improve_image(b_bytes) or b_bytes
                
                pdf_data = generate_cnic_pdf(f_bytes, b_bytes)
                
                st.download_button(
                    "📥 Download PDF",
                    data=pdf_data,
                    file_name="cnic_print.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()