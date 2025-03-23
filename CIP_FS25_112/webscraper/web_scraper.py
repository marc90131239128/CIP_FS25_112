from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from bs4 import BeautifulSoup

# Automatically install ChromeDriver
service = ChromeService(ChromeDriverManager().install())


chrome_options = Options() 
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")


driver = webdriver.Chrome(service=service, options=chrome_options)

# Base URL (First page has a different URL)
BASE_URL_FIRST_PAGE = "https://www.immoscout24.ch/de/immobilien/mieten/ort-st-gallen"
BASE_URL_PAGINATED = "https://www.immoscout24.ch/de/immobilien/mieten/ort-st-gallen?pn={}"

# List to store apartment data
apartments = []

# Function to extract text using CSS selectors
def get_text_css(driver, label, css_selector, multiple=False):
    """Extracts text from an element using a CSS selector."""
    try:
        if multiple:
            elements = driver.find_elements(By.CSS_SELECTOR, css_selector)
            text = ", ".join([el.text.strip() for el in elements if el.text.strip()])
        else:
            element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
            )
            text = element.text.strip()
        print(f"✅ {label}: {text}")
        return text
    except:
        print(f"⚠ {label} NOT FOUND")
        return "N/A"

try:
    page_number = 1  # Start with first page
    total_listings_scraped = 0  # Counter for total listings
    while True:
        #Handle first page separately (no ?pn=1 in URL)
        url = BASE_URL_FIRST_PAGE if page_number == 1 else BASE_URL_PAGINATED.format(page_number)

        print(f"🌍 Navigating to: {url}")
        driver.get(url)
        time.sleep(3)  # Wait for the page to load

        #Accept cookies if present
        try:
            accept_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            accept_button.click()
            print("✅ Cookie banner accepted!")
            time.sleep(2)
        except:
            print("⚠ No cookie banner found. Continuing...")

        #Wait for listings to load
        try:
            listings = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-test='result-list-item']"))
            )
            print(f"✅ Found {len(listings)} listings on page {page_number}")
            
            if len(listings) == 0:  # No more listings found
                print("🚀 No more listings found. Scraping completed.")
                break
                
            for i in range(len(listings)):  # Loop through all listings on the page
                print(f"🔍 Scraping listing {i+1} on page {page_number}")

                #Refresh the list of listings (as page reloads)
                listings = driver.find_elements(By.CSS_SELECTOR, "div[data-test='result-list-item']")
                listing = listings[i]

                #Click on the listing
                listing_link = listing.find_element(By.TAG_NAME, "a")
                listing_url = listing_link.get_attribute("href")
                driver.execute_script("arguments[0].scrollIntoView();", listing)
                time.sleep(1)
                listing_link.click()
                time.sleep(3)

                #Scrape details from the listing page using CSS selectors
                price = get_text_css(driver, "Price", "div.SpotlightAttributesPrice_value_TqKGz > span:nth-child(2)")
                net_price = get_text_css(driver, "Net Price", "dl > dd:nth-child(2) > span")
                nebenkosten = get_text_css(driver, "Additional Costs (Nebenkosten)", "dl > dd:nth-child(4) > span")
                living_space = get_text_css(driver, "Living Space", "div.SpotlightAttributesUsableSpace_value_cpfrh")
                rooms = get_text_css(driver, "Rooms", "div.SpotlightAttributesNumberOfRooms_value_TUMrd")
                street = get_text_css(driver, "Street", "span.AddressDetails_street_nXScL")
                plz_city = get_text_css(driver, "PLZ & City", "address > span:nth-child(2)")
                description = get_text_css(driver, "Description", "div.Description_descriptionBody_AYyuy")
                eigenschaften = get_text_css(driver, "Eigenschaften", "ul.FeaturesFurnishings_list_S54KV li", multiple=True)

                #Extract "Hauptangaben" as a dictionary of key-value pairs
                def get_hauptangaben(driver):
                    """Extracts all key-value pairs from 'Hauptangaben'."""
                    try:
                        labels = driver.find_elements(By.CSS_SELECTOR, "div.CoreAttributes_coreAttributes_e2NAm dl dt")
                        values = driver.find_elements(By.CSS_SELECTOR, "div.CoreAttributes_coreAttributes_e2NAm dl dd")
                        return {label.text.strip(): values[i].text.strip() for i, label in enumerate(labels)}
                    except:
                        return {}

                hauptangaben = get_hauptangaben(driver)
                print(f"✅ Hauptangaben: {hauptangaben}")

                #Store apartment details
                apartments.append({
                    "City": "Zurich",
                    "Price (CHF/Month)": price,
                    "Net Price": net_price,
                    "Additional Costs": nebenkosten,
                    "Rooms": rooms,
                    "Living Space": living_space,
                    "Street": street,
                    "PLZ/City": plz_city,
                    "Description": description,
                    "Eigenschaften": eigenschaften,
                    "Hauptangaben": hauptangaben,
                    "Listing URL": listing_url
                })

                #Go back to the listings page
                driver.back()
                time.sleep(3)  # Allow the page to reload

            # After scraping all listings on the page
            total_listings_scraped += len(listings)
            print(f"Total listings scraped: {total_listings_scraped}")
            
            # Move to next page
            page_number += 1
            print(f"Moving to page {page_number}")
            
        except:
            print("No more listings found or error occurred. Scraping completed.")
            break

finally:
    driver.quit()  # Close WebDriver
    print("Browser closed.")

# Save results to a DataFrame
df2 = pd.DataFrame(apartments)
df2.to_csv("immoscout_st.gallen_1_onwards.csv", index=False)
print(" Data saved to immoscout_st.gallen_1_onwards.csv")

# Display results
print(df2.head())








