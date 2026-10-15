from selenium import webdriver
from selenium.webdriver.common.by import By


def test_application():

    driver = webdriver.Chrome()

    try:

        driver.get("http://localhost:8000/docs")

        assert "Swagger UI" in driver.title

    finally:

        driver.quit()
