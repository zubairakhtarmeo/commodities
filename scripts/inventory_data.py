from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import sys
import time
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from selenium.common.exceptions import TimeoutException


def _load_oracle_credentials() -> tuple[str, str]:
    """Load Oracle Fusion credentials without hardcoding them.

    Resolution order:
      1. ORACLE_USERNAME / ORACLE_PASSWORD environment variables
         (injected automatically by GitHub Actions from repository secrets)
      2. .streamlit/secrets.toml under the project root
         (local development only — this file is gitignored)

    Exits with a clear error message if either credential is missing.
    """
    oracle_user = os.environ.get("ORACLE_USERNAME", "").strip()
    oracle_pass = os.environ.get("ORACLE_PASSWORD", "").strip()

    if not oracle_user or not oracle_pass:
        secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                try:
                    import tomllib                    # Python 3.11+
                except ImportError:
                    import tomli as tomllib           # pip install tomli on Python < 3.11
                with open(secrets_path, "rb") as _f:
                    _s = tomllib.load(_f)
                oracle_user = oracle_user or _s.get("ORACLE_USERNAME", "").strip()
                oracle_pass = oracle_pass or _s.get("ORACLE_PASSWORD", "").strip()
            except Exception as _e:
                print(f"Warning: could not read secrets.toml: {_e}")

    if not oracle_user:
        print(
            "FATAL: ORACLE_USERNAME is not set.\n"
            "  • GitHub Actions: add ORACLE_USERNAME to repository secrets.\n"
            "  • Local dev: add ORACLE_USERNAME = '...' to .streamlit/secrets.toml."
        )
        sys.exit(1)
    if not oracle_pass:
        print(
            "FATAL: ORACLE_PASSWORD is not set.\n"
            "  • GitHub Actions: add ORACLE_PASSWORD to repository secrets.\n"
            "  • Local dev: add ORACLE_PASSWORD = '...' to .streamlit/secrets.toml."
        )
        sys.exit(1)

    return oracle_user, oracle_pass

# ================= Config =================
base_directory = r"D:\path"
os.makedirs(base_directory, exist_ok=True)
report_name = "MG_STOCK_TILL_DATE_MULTIPLE_UNIT"
PROCESS_NAME = "MG_STOCK_TILL_DATE_MULTIPLE_UNIT"
unit_name = "MG Apparel Unit 2"


# Create download path
download_directory = base_directory
os.makedirs(download_directory, exist_ok=True)

expected_filename = "MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx"
target_file_path = os.path.join(download_directory, expected_filename)


# ================= Browser Setup =================
chrome_options = webdriver.ChromeOptions()
prefs = {
    "download.default_directory": download_directory.replace('/', '\\'),
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "profile.default_content_setting_values.automatic_downloads": 1,
}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--start-maximized")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 30)

# ================= Login =================
url = "https://fa-exeu-saasfaprod1.fa.ocs.oraclecloud.com/fscmUI/faces/FuseOverview?_adf.ctrl-state=vxzfoxjw4_5&fndGlobalItemNodeId=itemNode_tools_scheduled_processes_fuse_plus"
driver.get(url)

username = wait.until(EC.presence_of_element_located((By.ID, "idcs-signin-basic-signin-form-username")))
password = driver.find_element(By.ID, "idcs-signin-basic-signin-form-password|input")
login_button = driver.find_element(By.ID, "idcs-signin-basic-signin-form-submit")

oracle_user, oracle_pass = _load_oracle_credentials()
username.send_keys(oracle_user)
time.sleep(2)
password.send_keys(oracle_pass)
time.sleep(2)
login_button.click()
print("Logged in successfully.")
time.sleep(5)

# ================= Start New Process =================
schedule_button = wait.until(EC.element_to_be_clickable(
    (By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:panel:scheduleProcess")
))
schedule_button.click()
print("Clicked 'Schedule New Process'.")
time.sleep(3)

# ================= Enter Process Name =================
name_field = wait.until(EC.presence_of_element_located(
    (By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:selectOneChoice2::content")
))
time.sleep(2)
name_field.send_keys(PROCESS_NAME)

name_field.send_keys(Keys.RETURN)
print("Entered process name.")
time.sleep(3)

# Click OK
ok_button = wait.until(EC.element_to_be_clickable(
    (By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:snpokbtnid")
))
ok_button.click()
print("Clicked OK.")
time.sleep(5)

# ================= Enter Parameters =================
# Entering the last date of most recent month
date_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute1::content"
)))
# Calculate last day of previous month
last_day_prev_month = (datetime.today().replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
#remove whole field first
date_field.clear()
date_field.send_keys(last_day_prev_month)
date_field.send_keys(Keys.RETURN)
print(f"Entered date: {last_day_prev_month}")
time.sleep(3)


# Business Unit- "MSM Spinning U1 - Raw Material"
bu_1="MSM - Spinning U1 - Raw Material"
bu_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute9::content"
)))
time.sleep(1)
bu_field.send_keys(bu_1)
bu_field.send_keys(Keys.RETURN)
print(f"Entered Business Unit: {bu_1}")
time.sleep(3)

# Business Unit- "MSM Spinning U2 - Raw Material"
bu_2="MSM - Spinning U2 - Raw Material"
bu_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute10::content"
)))
time.sleep(1)
bu_field.send_keys(bu_2)
bu_field.send_keys(Keys.RETURN)
print(f"Entered Business Unit: {bu_2}")
time.sleep(3)

# Business Unit- "MSL Fiber U3 - Raw Material"
bu_3="MSL - Fibres U3 - Raw Material"
bu_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute11::content"
)))
time.sleep(1)
bu_field.send_keys(bu_3)
bu_field.send_keys(Keys.RETURN)
print(f"Entered Business Unit: {bu_3}")
time.sleep(3)

# Business Unit- "MTM Spinning U1 - Raw Material"
bu_4="MTM - Spinning U1 - Raw Material"
bu_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute12::content"
)))
time.sleep(1)
bu_field.send_keys(bu_4)
bu_field.send_keys(Keys.RETURN)
print(f"Entered Business Unit: {bu_4}")
time.sleep(3)

# Business Unit- "MTM Spinning U2 - Raw Material"
bu_5="MTM - Spinning U2 - Raw Material"
bu_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute13::content"
)))
time.sleep(1)
bu_field.send_keys(bu_5)
bu_field.send_keys(Keys.RETURN)
print(f"Entered Business Unit: {bu_5}")
time.sleep(3)

# Business Unit- "MTM Spinning U3 - Raw Material"
bu_6="MTM - Spinning U3 - Raw Material"
bu_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute14::content"
)))
time.sleep(1)
bu_field.send_keys(bu_6)
bu_field.send_keys(Keys.RETURN)
print(f"Entered Business Unit: {bu_6}")
time.sleep(3)

# Business Unit- "MTM Spinning U5 - Raw Material"
bu_7="MTM - Spinning U5 - Raw Material"
bu_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute15::content"
)))
time.sleep(1)
bu_field.send_keys(bu_7)
bu_field.send_keys(Keys.RETURN)
print(f"Entered Business Unit: {bu_7}")
time.sleep(3)

# Business Unit- "MTM Spinning U6 - Raw Material"
bu_8="MTM - Spinning U6 - Raw Material"
bu_field = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:basicReqBody:paramDynForm_Attribute16::content"
)))
time.sleep(1)
bu_field.send_keys(bu_8)
bu_field.send_keys(Keys.RETURN)
print(f"Entered Business Unit: {bu_8}")
time.sleep(3)

# ================= Submit =================
submit_button = wait.until(EC.element_to_be_clickable((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:requestBtns:submitButton"
)))
submit_button.click()
print("Clicked Submit.")
time.sleep(3)

# Capture Process ID
confirmation = wait.until(EC.presence_of_element_located((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:requestBtns:confirmationPopup:pt_ol1"
)))
process_id = ''.join(filter(str.isdigit, confirmation.text))
print(f"Captured Process ID: {process_id}")

ok_button = wait.until(EC.element_to_be_clickable((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:r1:0:r1:requestBtns:confirmationPopup:confirmSubmitDialog::ok"
)))
ok_button.click()
print("Closed confirmation popup.")
time.sleep(15)

# ================= Monitor Process =================
expand_search = wait.until(EC.element_to_be_clickable((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:srRssdfl::_afrDscl"
)))
expand_search.click()
print("Expanded search panel.")
time.sleep(2)

# Enter Process ID in correct field
pid_input = wait.until(EC.presence_of_element_located((
    By.XPATH, "//label[contains(text(),'Process ID')]/following::input[1]"
)))
pid_input.clear()
pid_input.send_keys(process_id)
print(f"Entered Process ID: {process_id}")
time.sleep(1)

# Click Search
search_button = wait.until(EC.element_to_be_clickable((
    By.ID, "_FOpt1:_FOr1:0:_FONSr2:0:_FOTsr1:0:pt1:srRssdfl::search"
)))
search_button.click()
print("Searched by Process ID.")
time.sleep(5)

# ===== Keep refreshing until process succeeds =====

max_duration = 600
refresh_interval = 10
start_time = time.time()

# Generic XPath for process result row
status_xpath = (
    "//table[contains(@summary,'List of Processes')]"
    "//tr[contains(., 'MG_STOCK_TILL_DATE_MULTIPLE_UNIT')]"
)

while time.time() - start_time < max_duration:

    try:
        # Re-locate row every loop
        status_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.XPATH, status_xpath)
            )
        )

        # Read full row text
        status_text = driver.execute_script(
            "return arguments[0].innerText;",
            status_element
        ).strip()

        print(f"Current Status => [{status_text}]")

        # ===== STATUS CHECKS =====
        normalized_status = status_text.lower()

        if "succeeded" in normalized_status:
            print(f"Process {process_id} succeeded.")
            break

        elif "error" in normalized_status:
            raise Exception(f"Process {process_id} failed with ERROR.")

        elif "warning" in normalized_status:
            print(f"Process {process_id} completed with WARNING.")
            break

        else:
            print("Process still running...")

    except Exception as e:
        print(f"Status read error: {e}")

    # ===== REFRESH =====
    try:
        refresh_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//img[contains(@title,'Refresh')]"
            ))
        )

        driver.execute_script(
            "arguments[0].click();",
            refresh_button
        )

        print("Clicked on Refresh button.")

    except Exception as e:
        print(f"Refresh failed: {e}")

    # Wait before next status check
    time.sleep(refresh_interval)

else:
    raise TimeoutException(
        f"Process {process_id} did not complete within timeout."
    )


# ================= Republish & Export =================
time.sleep(5)
mg_report_td = wait.until(EC.element_to_be_clickable(
    (By.XPATH, "//td[contains(@class, 'xen') and contains(., 'MG_STOCK_TILL_DATE_MULTIPLE_UNIT')]")
))
mg_report_td.click()
print("Clicked on the 'MG_STOCK_TILL_DATE_MULTIPLE_UNIT' cell.")
time.sleep(5)

# Republish Online
iframes = driver.find_elements(By.TAG_NAME, "iframe")
republish_button = None
for iframe in iframes:
    driver.switch_to.frame(iframe)
    try:
        republish_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//img[contains(@src, 'viewreport_ena.png')]")
        ))
        break
    except:
        driver.switch_to.default_content()

if republish_button:
    driver.execute_script("arguments[0].scrollIntoView();", republish_button)
    time.sleep(2)
    republish_button.click()
    print("Clicked on 'Republish Online' button.")
driver.switch_to.default_content()

time.sleep(5)
driver.switch_to.window(driver.window_handles[-1])
print("Switched to new window.")

popup_menu = wait.until(EC.element_to_be_clickable((
    By.XPATH, "//img[contains(@src, 'toolbar/popupmenu_ena.png')]"
)))
popup_menu.click()
print("Clicked on 'Popup Menu' button.")

export_button = wait.until(EC.element_to_be_clickable((
    By.XPATH, "//a[contains(@class, 'masterMenuItem') and .//div[contains(text(), 'Export')]]"
)))
driver.execute_script("arguments[0].scrollIntoView();", export_button)
time.sleep(2)
export_button.click()
print("Clicked on 'Export' button.")

excel_option = wait.until(EC.element_to_be_clickable((
    By.XPATH, "//a[contains(@class, 'masterMenuItem') and .//div[contains(text(), 'Excel (*.xlsx)')]]"
)))
driver.execute_script("arguments[0].scrollIntoView();", excel_option)
time.sleep(1)
ActionChains(driver).move_to_element(excel_option).click().perform()
print("Clicked on 'Excel (*.xlsx)' option.")

# ================= Download Monitoring =================

print("Waiting for new file to appear in download directory...")
before_files = set(os.listdir(download_directory))

downloaded_file = None
max_wait = 120  # seconds
elapsed = 0

while elapsed < max_wait:
    time.sleep(2)
    elapsed += 2
    after_files = set(os.listdir(download_directory))
    new_files = after_files - before_files
    new_files = [f for f in new_files if f.endswith(".xlsx") and not f.endswith(".crdownload")]
    if new_files:
        downloaded_file = os.path.join(download_directory, new_files[0])
        print(f"New file detected: {new_files[0]}")
        break
    else:
        print("Waiting for download...")

expected_filename = "MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx"
target_file_path = os.path.join(download_directory, expected_filename)

if not downloaded_file:
    print(" Download did not complete in time. Keeping old file untouched.")
else:
    try:
        # If old file exists, delete it ONLY after new file is ready
        if os.path.exists(target_file_path):
            os.remove(target_file_path)
            print(f"Deleted old file: {expected_filename}")

        os.rename(downloaded_file, target_file_path)
        print(f"Renamed new file '{os.path.basename(downloaded_file)}' to '{expected_filename}'")
        print(f" Download completed successfully → {target_file_path}")
    except Exception as e:
        print(f"Error renaming new file: {e}")
 