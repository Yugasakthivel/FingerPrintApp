import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from datetime import datetime
import os

LOG_FILE = "auth_logs.csv"

# ---- Initialize Log File ----
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["User", "Status", "Timestamp"]).to_csv(LOG_FILE, index=False)

# ---- Compare Fingerprints ----
def compare_fingerprints(uploaded_img, stored_img):
    uploaded_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)
    stored_gray = cv2.cvtColor(stored_img, cv2.COLOR_BGR2GRAY)

    uploaded_gray = cv2.resize(uploaded_gray, (200, 200))
    stored_gray = cv2.resize(stored_gray, (200, 200))

    diff = np.sum((uploaded_gray.astype("float") - stored_gray.astype("float")) ** 2)
    mse = diff / float(uploaded_gray.shape[0] * uploaded_gray.shape[1])
    return mse

# ---- Log Attempts ----
def log_attempt(user, status):
    df = pd.read_csv(LOG_FILE)
    new_entry = {"User": user, "Status": status, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

# ---- Reset Logs ----
def reset_logs():
    pd.DataFrame(columns=["User", "Status", "Timestamp"]).to_csv(LOG_FILE, index=False)

# ---- User Database (templates) ----
fingerprint_db = {
    "Alice": "c:/StreamlitApps/FingerPrintApp/templates/alice.png",
    "Bob": "c:/StreamlitApps/FingerPrintApp/templates/bob.png",
    "Charlie": "c:/StreamlitApps/FingerPrintApp/templates/charlie.png"
}

# ---- Streamlit UI ----
st.set_page_config(page_title="Fingerprint Authentication System", layout="wide")
st.title("🔒 Fingerprint Authentication System")

selected_user = st.selectbox("Select User", list(fingerprint_db.keys()))
uploaded_file = st.file_uploader("📂 Upload Fingerprint Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_img = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_img, caption="Uploaded Fingerprint", use_container_width=True)

    if st.button("Authenticate"):
        stored_img = cv2.imread(fingerprint_db[selected_user])

        if stored_img is None:
            st.error(f"⚠️ Stored fingerprint not found for {selected_user}")
        else:
            mse = compare_fingerprints(uploaded_img, stored_img)

            if mse < 1000:  # threshold
                status = "Success"
                st.success(f"✅ Authentication Success for {selected_user}")
            else:
                status = "Failed"
                st.error(f"❌ Authentication Failed for {selected_user}")

            log_attempt(selected_user, status)

# ---- Show Logs ----
st.subheader("📋 Authentication Logs")
df = pd.read_csv(LOG_FILE)
st.dataframe(df, use_container_width=True)

# ---- Download + Reset Buttons ----
if not df.empty:
    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Logs as CSV",
        data=csv,
        file_name="auth_logs.csv",
        mime="text/csv"
    )

    # Excel download
    excel_file = "auth_logs.xlsx"
    df.to_excel(excel_file, index=False)
    with open(excel_file, "rb") as f:
        st.download_button(
            label="⬇️ Download Logs as Excel",
            data=f,
            file_name="auth_logs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Reset logs button
    if st.button("🗑 Reset Logs"):
        reset_logs()
        st.warning("Logs have been cleared. Refresh the page.")

# ---- Dashboard / Analytics ----
if not df.empty:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    col1, col2 = st.columns(2)

    # Success vs Fail
    with col1:
        st.subheader("✅ Success vs ❌ Fail")
        fig1, ax1 = plt.subplots()
        df["Status"].value_counts().plot(kind="bar", ax=ax1, color=["green", "red"])
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Status")
        st.pyplot(fig1)

    # Attempts Over Time
    with col2:
        st.subheader("📈 Attempts Over Time")
        attempts_time = df.groupby(df["Timestamp"].dt.date).size()
        fig2, ax2 = plt.subplots()
        attempts_time.plot(kind="line", marker="o", ax=ax2)
        ax2.set_ylabel("Attempts")
        ax2.set_xlabel("Date")
        st.pyplot(fig2)

    # Top Active Users
    st.subheader("👥 Most Active Users")
    fig3, ax3 = plt.subplots()
    df["User"].value_counts().plot(kind="bar", ax=ax3, color="blue")
    ax3.set_ylabel("Login Count")
    st.pyplot(fig3)

    # Heatmap of Activity
    st.subheader("🔥 Login Activity Heatmap")
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day_name()
    pivot = pd.pivot_table(df, values="User", index="Day", columns="Hour", aggfunc="count", fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt="d", ax=ax4)
    st.pyplot(fig4)

else:
    st.info("No authentication logs yet. Try uploading and authenticating a fingerprint.")
