import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

topic = input("Enter the topic you want to search for: ")

driver = webdriver.Chrome()
driver.get("https://finance.yahoo.com/quote/" + topic + "/history")

data = []

try:
    rows = WebDriverWait(driver, 30).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tr.yf-1jecxey"))
    )
    print("Found", len(rows), "rows")
    for row in rows:
        try:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) == 7:
                date, open_, high, low, close, adj_close, volume = [cell.text for cell in cells]
                data.append({
                    "Date": date,
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Adj Close": adj_close,
                    "Volume": volume
                })
            else:
                continue
        except Exception as e:
            print("Error extracting row", e)
            continue
except Exception as e:
    print("No rows found:", e)
finally:
    driver.quit()

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv(f"historical_data.csv", index=False)
print(f"Saved data to {topic}_historical_data.csv")



