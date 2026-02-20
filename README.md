# 🛡️ Secure Geofenced Face-Recognition Attendance System

An enterprise-grade attendance solution that combines Biometric Authentication, Geofencing, and Google Workspace integration to eliminate proxy attendance and manual errors.

## 🚀 Key Features
- **📧 Email-Restricted Access:** Only authorized office email IDs can log into the portal.
- **👤 Identity Verification:** Real-time face recognition (Python/OpenCV) ensures the person logged in matches the person on camera.
- **📍 Smart Geofencing:** Attendance can only be marked within a **200m - 400m radius** of 4 predefined office locations.
- **☁️ Serverless Data Logging:** Attendance records are synced in real-time to **Google Sheets**, and user registration photos are stored in **Google Drive**.
- **🚫 Anti-Proxy Logic:** Prevents duplicate punch-ins and ensures physical presence.

## 💻 Tech Stack
- **Frontend:** HTML5, CSS3, JavaScript (Geolocation API)
- **Backend Logic:** Python (OpenCV, Face_Recognition library)
- **Database/Storage:** Google Sheets API, Google Drive API
- **Scripting:** Google Apps Script (for seamless data flow)

## 📸 Screenshots
| Face Unlock |

<img width="1072" height="1036" alt="image" src="https://github.com/user-attachments/assets/80506ea0-b432-44ce-8fda-cea47dd73f19" />
<img width="1072" height="1036" alt="image" src="https://github.com/user-attachments/assets/e0ad643e-8ca7-42db-9f7a-f4d1f288ae5a" />


| Attendance Portal |


<img width="595" height="817" alt="image" src="https://github.com/user-attachments/assets/de48d9d6-96ff-4cc6-be27-acf4d7527b52" />
<img width="503" height="793" alt="image" src="https://github.com/user-attachments/assets/8f3307e6-72c0-4767-8c76-1c5a4dac6ea3" />



| Data in Google Sheets |

<img width="784" height="382" alt="image" src="https://github.com/user-attachments/assets/c20a92ba-4088-4b8b-9f47-89d3c5d945a0" />
<img width="1892" height="523" alt="image" src="https://github.com/user-attachments/assets/6b72ca48-0a12-4ec2-90a2-74d8a15f9b67" />

## 🛠️ Installation & Setup
1. Clone the repository: `git clone [Your-Repo-Link]`
2. Install Python dependencies: `pip install opencv-python face-recognition gspread oauth2client`
3. Set up your Google Cloud Console credentials and download `credentials.json`.
4. Run the Python authentication script: `python app.py`
5. Open `index.html` to access the web portal.

## 📐 System Architecture
`User Login -> Face Scan -> Geolocation Check -> Attendance Action -> Google Sheets Update`
