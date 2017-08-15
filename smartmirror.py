# smartmirror.py
# requirements
# requests, feedparser, traceback, Pillow

from Tkinter import *
import locale
import threading
import time
import requests
from requests import get
import json
import traceback
import feedparser

from PIL import Image, ImageTk
from contextlib import contextmanager

import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import sys
import subprocess

import atexit

def exit_handler():
    print 'Application is ending!'
    print 'Cleaning Up:'
    print 'Setting display_power to 1...'
    subprocess.call(["vcgencmd display_power 1"], shell=True)
    print 'Success: display_power = 1!'




LOCALE_LOCK = threading.Lock()

ip = '150.250.125.232'
ui_locale = '' # e.g. 'fr_FR' fro French, '' as default
time_format = 12 # 12 or 24
date_format = "%b %d, %Y" # check python doc for strftime() for options
news_country_code = 'us'
weather_api_token = '83afd8c71153c1996fb93ff670b33366' # create account at https://darksky.net/dev/
weather_lang = 'en' # see https://darksky.net/dev/docs/forecast for full list of language parameters values
weather_unit = 'us' # see https://darksky.net/dev/docs/forecast for full list of unit parameters values
latitude = None # Set this if IP location lookup does not work for you (must be a string)
longitude = None # Set this if IP location lookup does not work for you (must be a string)
xlarge_text_size = 96
large_text_size = 48
medium_text_size = 28
small_text_size = 18
xsmall_text_Size = 13

@contextmanager
def setlocale(name): #thread proof function to work with locale
    with LOCALE_LOCK:
        saved = locale.setlocale(locale.LC_ALL)
        try:
            yield locale.setlocale(locale.LC_ALL, name)
        finally:
            locale.setlocale(locale.LC_ALL, saved)

# maps open weather icons to
# icon reading is not impacted by the 'lang' parameter
icon_lookup = {
    'clear-day': "assets/Sun.png",  # clear sky day
    'wind': "assets/Wind.png",   #wind
    'cloudy': "assets/Cloud.png",  # cloudy day
    'partly-cloudy-day': "assets/PartlySunny.png",  # partly cloudy day
    'rain': "assets/Rain.png",  # rain day
    'snow': "assets/Snow.png",  # snow day
    'snow-thin': "assets/Snow.png",  # sleet day
    'fog': "assets/Haze.png",  # fog day
    'clear-night': "assets/Moon.png",  # clear sky night
    'partly-cloudy-night': "assets/PartlyMoon.png",  # scattered clouds night
    'thunderstorm': "assets/Storm.png",  # thunderstorm
    'tornado': "assests/Tornado.png",    # tornado
    'hail': "assests/Hail.png"  # hail
}

class Clock(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        # initialize time label
        self.time1 = ''
        self.timeLbl = Label(self, font=('Helvetica', large_text_size), fg="white", bg="black")
        self.timeLbl.pack(side=TOP, anchor=E)
        # initialize day of week
        self.day_of_week1 = ''
        self.dayOWLbl = Label(self, text=self.day_of_week1, font=('Helvetica', small_text_size), fg="white", bg="black")
        self.dayOWLbl.pack(side=TOP, anchor=E)
        # initialize date label
        self.date1 = ''
        self.dateLbl = Label(self, text=self.date1, font=('Helvetica', small_text_size), fg="white", bg="black")
        self.dateLbl.pack(side=TOP, anchor=E)
        self.tick()

    def tick(self):
        with setlocale(ui_locale):
            if time_format == 12:
                time2 = time.strftime('%I:%M %p') #hour in 12h format
            else:
                time2 = time.strftime('%H:%M') #hour in 24h format

            day_of_week2 = time.strftime('%A')
            date2 = time.strftime(date_format)
            # if time string has changed, update it
            if time2 != self.time1:
                self.time1 = time2
                self.timeLbl.config(text=time2)
            if day_of_week2 != self.day_of_week1:
                self.day_of_week1 = day_of_week2
                self.dayOWLbl.config(text=day_of_week2)
            if date2 != self.date1:
                self.date1 = date2
                self.dateLbl.config(text=date2)
            # calls itself every 200 milliseconds
            # to update the time display as needed
            # could use >200 ms, but display gets jerky
            self.timeLbl.after(200, self.tick)

class Weather(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, bg='black')
        self.temperature = ''
        self.forecast = ''
        self.location = ''
        self.currently = ''
        self.icon = ''
        self.degreeFrm = Frame(self, bg="black")
        self.degreeFrm.pack(side=TOP, anchor=W)
        self.temperatureLbl = Label(self.degreeFrm, font=('Helvetica', xlarge_text_size), fg="white", bg="black")
        self.temperatureLbl.pack(side=LEFT, anchor=N)
        self.iconLbl = Label(self.degreeFrm, bg="black")
        self.iconLbl.pack(side=LEFT, anchor=N, padx=20)
        self.currentlyLbl = Label(self, font=('Helvetica', medium_text_size), fg="white", bg="black")
        self.currentlyLbl.pack(side=TOP, anchor=W)
        self.forecastLbl = Label(self, font=('Helvetica', small_text_size), wraplength=600, justify=LEFT, fg="white", bg="black")
        self.forecastLbl.pack(side=TOP, anchor=W)
        self.locationLbl = Label(self, font=('Helvetica', small_text_size), fg="white", bg="black")
        self.locationLbl.pack(side=TOP, anchor=W)
        self.after(5)
        self.get_weather()

    def get_ip(self):
        try:
            ip_url = 'https://api.ipify.org'
            ip = get(ip_url).text
            #req = requests.get(ip_url)
            #ip_json = json.loads(req.text)
            print ip
            return ip
        except Exception as e:
            traceback.print_exc()
            return "Error: %s. Cannot get ip :148." % str(e)

    def get_weather(self):
        try:

            if latitude is None and longitude is None:
                # get location
                location_req_url = "http://freegeoip.net/json/%s" % ''#self.get_ip()
                r = requests.get(location_req_url)
                location_obj = json.loads(r.text)

                lat = location_obj['latitude']
                lon = location_obj['longitude']

                location2 = "%s, %s" % (location_obj['city'], location_obj['region_code'])

                # get weather
                weather_req_url = "https://api.darksky.net/forecast/%s/%s,%s?lang=%s&units=%s" % (weather_api_token, lat,lon,weather_lang,weather_unit)
            else:
                location2 = ""
                # get weather
                weather_req_url = "https://api.darksky.net/forecast/%s/%s,%s?lang=%s&units=%s" % (weather_api_token, latitude, longitude, weather_lang, weather_unit)
            r = requests.get(weather_req_url)
            weather_obj = json.loads(r.text)

            degree_sign= u'\N{DEGREE SIGN}'
            temperature2 = "%s%s" % (str(int(weather_obj['currently']['temperature'])), degree_sign)
            currently2 = weather_obj['currently']['summary']
            forecast2 = weather_obj["hourly"]["summary"]

            icon_id = weather_obj['currently']['icon']
            icon2 = None

            if icon_id in icon_lookup:
                icon2 = icon_lookup[icon_id]

            if icon2 is not None:
                if self.icon != icon2:
                    self.icon = icon2
                    image = Image.open(icon2)
                    image = image.resize((100, 100), Image.ANTIALIAS)
                    image = image.convert('RGB')
                    photo = ImageTk.PhotoImage(image)

                    self.iconLbl.config(image=photo)
                    self.iconLbl.image = photo
            else:
                # remove image
                self.iconLbl.config(image='')

            if self.currently != currently2:
                self.currently = currently2
                self.currentlyLbl.config(text=currently2)
            if self.forecast != forecast2:
                self.forecast = forecast2
                self.forecastLbl.config(text=forecast2)
            if self.temperature != temperature2:
                self.temperature = temperature2
                self.temperatureLbl.config(text=temperature2)
            if self.location != location2:
                if location2 == ", ":
                    self.location = "Cannot Pinpoint Location"
                    self.locationLbl.config(text="Cannot Pinpoint Location")
                else:
                    self.location = location2
                    self.locationLbl.config(text=location2)
        except Exception as e:
            traceback.print_exc()
            print "Error: %s. Cannot get weather." % e

        self.after(600000, self.get_weather)

    @staticmethod
    def convert_kelvin_to_fahrenheit(kelvin_temp):
        return 1.8 * (kelvin_temp - 273) + 32

class News(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.config(bg='black')
        self.title = 'News' # 'News' is more internationally generic
        self.newsLbl = Label(self, text=self.title, font=('Helvetica', medium_text_size), fg="white", bg="black")
        self.newsLbl.pack(side=TOP, anchor=W)
        self.headlinesContainer = Frame(self, bg="black")
        self.headlinesContainer.pack(side=TOP)
        self.after(5)
        self.get_headlines()

    def get_headlines(self):
        try:
            # remove all children
            for widget in self.headlinesContainer.winfo_children():
                widget.destroy()
            if news_country_code == None:
                headlines_url = "https://news.google.com/news?ned=us&output=rss"
            else:
                headlines_url = "https://news.google.com/news?ned=%s&output=rss" % news_country_code
            feed = feedparser.parse(headlines_url)
            for post in feed.entries[0:5]:
                headline = NewsHeadline(self.headlinesContainer, post.title)
                headline.pack(side=TOP, anchor=W)
        except Exception as e:
            traceback.print_exc()
            print "Error: %s. Cannot get news." % e

        self.after(600000, self.get_headlines)

class NewsHeadline(Frame):
    def __init__(self, parent, event_name=""):
        Frame.__init__(self, parent, bg='black')

        image = Image.open("assets/Newspaper.png")
        image = image.resize((20, 20), Image.ANTIALIAS)
        image = image.convert('RGB')
        photo = ImageTk.PhotoImage(image)

        self.iconLbl = Label(self, bg='black', image=photo)
        self.iconLbl.image = photo
        self.iconLbl.pack(side=LEFT, anchor=N)

        self.eventName = event_name[0:91]
        self.eventNameLbl = Label(self, text=self.eventName, font=('Helvetica', xsmall_text_Size), fg="white", bg="black")
        self.eventNameLbl.pack(side=LEFT, anchor=N)

class FacialRec(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.config(bg='black')        
        self.e2fRatio_min = 0.2142
        self.e2fRatio_max = 0.5
        self.f2pRatio_min = 0.1
        self.f2pRatio_max = 1.0
        self.f2mRatio_min = 0.245
        self.f2sRatio_min = 0.245
        self.isRecognizedFace = False
        self.emotionThreshold = 5
        self.maxSmileCountdown = 3
        self.smileCountdown = 0
        
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier('mouth.xml')
        self.smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
        
        self.camera = PiCamera()
        self.camera.resolution = (128,160)
        self.faceLbl = Label(self)
        self.facialRecognition()
        self.faceLbl.config(image=self.imgtk)
        self.faceLbl.pack()        
    
    def facePic(self):
        rawCapture = PiRGBArray(self.camera)
        #time.sleep(0.3)
        self.camera.capture(rawCapture, format="bgr")
        img = rawCapture.array
        #img = cv2.resize(img, (240, 320))
        # Rearrang the color channel
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))

        # Convert the Image object into a TkPhoto object
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        print "facePic() ran"
        return img

    def facialRecognition(self):
        rawCapture = PiRGBArray(self.camera)

        # allow the camera to warmup
        #time.sleep(0.3)

        self.camera.capture(rawCapture, format="bgr")

        img = rawCapture.array
        #img = cv2.resize(img, (120, 160))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find face 
        pheight, pwidth, pchannels = img.shape
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1, minSize=(int(pheight * self.f2pRatio_min), int(pwidth * self.f2pRatio_min)),maxSize=(int(pheight * self.f2pRatio_max), int(pwidth * self.f2pRatio_max)),  flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        self.isRecognizedFace = len(faces) != 0

        if (self.isRecognizedFace):
            print "face found!"
            
            # iterate through faces
            for (x, y, w, h) in faces:
                # draw faces on img
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                
                # find eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray, minSize=(int(self.e2fRatio_min * w), int(self.e2fRatio_min * h)), maxSize=(int(self.e2fRatio_max * w), int(self.e2fRatio_max * h)))
                # iterate through eyes
                if(len(eyes)>0):
                        for (ex, ey, ew, eh) in eyes[0:2]:
                            # draw eyes on img
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                else:
                    ex,ey,ew,eh = 0,0,0,0
                # find mouths
                mouths = self.mouth_cascade.detectMultiScale(roi_gray, minSize=(int(self.f2mRatio_min * w),int(self.f2mRatio_min *h)))
                if len(mouths)>0:
                    mx, my, mw, mh=mouths[-1]
                    if my <= ey + eh:
                        print "no mouths in hindsight"
                    else:
                        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                else:
                    mx,my,mw,mh=0,0,0,0

                # find smiles
                smiles=self.smile_cascade.detectMultiScale(roi_color,minSize=(int(self.f2sRatio_min * w), int(self.f2sRatio_min * h)))

                if len (smiles) >0:
                    sx,sy,sw,sh=smiles[-1]
                    if sy <= ey + eh:
                        print "no smiles in hindsight"
                    else:
                        # draw mouths on img
                        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
                else:
                    sx,sy,sw,sh=0,0,0,0
        	
        else:
            print "no faces found :("
        
        # cv2.imwrite('facetest.png',img)
        # Rearrang the color channel
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))

        # Convert the Image object into a TkPhoto object
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        print "facialRecognition ran"
        self.faceLbl.config(image=img)
        self.imgtk = img
        self.after(5, self.facialRecognition)

class FullscreenWindow:

    def __init__(self):
        self.tk = Tk()
        self.tk.configure(background='black')
        
        self.topFrame = Frame(self.tk, background = 'black')
        self.bottomFrame = Frame(self.tk, background = 'black')
        self.topFrame.pack(side = TOP, fill=BOTH, expand = YES)
        self.bottomFrame.pack(side = BOTTOM, fill=BOTH, expand = YES)

        self.state = True
        self.toggle_fullscreen
        self.tk.attributes("-fullscreen", self.state)
        self.tk.config(cursor='none')
        self.tk.bind("<Return>", self.toggle_fullscreen)
        self.tk.bind("<Escape>", self.end_fullscreen)
        
        # weather
        self.weather = Weather(self.topFrame)
        self.weather.pack(side=LEFT, anchor=N, padx = 10, pady=60)

        # clock
        self.clock = Clock(self.topFrame)
        self.clock.pack(side=RIGHT, anchor=N, padx = 20 ,pady=50)
        
        # facialRec
        self.facialRec = FacialRec(self.bottomFrame, padx = 40, pady = 50)
        self.facialRec.pack(side=RIGHT, anchor=S)
        
        # news
        self.news = News(self.bottomFrame)
        self.news.pack(side=LEFT, anchor=S, padx = 10, pady=50)


    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.tk.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.tk.attributes("-fullscreen", False)
        return "break"

class Manager():
    def __init__(self):
        self.w = FullscreenWindow()
        self.sleepTimer = 60#000000000000000000000 #should be 60 by default
        self.countDown=self.sleepTimer
        self.counter = 0
        self.w.tk.after(1, self.Manage)
        self.w.tk.mainloop()
        
    def Manage(self):
        print "countDown = %d" % self.countDown
        if(self.w.facialRec.isRecognizedFace):
            if(self.countDown <=0):
                subprocess.call(["vcgencmd display_power 1"], shell=True)
            self.countDown = self.sleepTimer
            self.counter = 0
        else:
            self.countDown -= 1

            if(self.countDown <= 0):
                self.counter+=1
                if(self.counter < 2):
                    subprocess.call(["vcgencmd display_power 0"], shell=True)
                else:
                    self.counter = 2
                self.countDown = 0
        self.w.tk.after(60, self.Manage)

if __name__ == '__main__':
    atexit.register(exit_handler)
    man = Manager()
