import pyrebase

config = {
    "apiKey": "AIzaSyCXHNMiEv2ma9GBPO30tqclx3YzGq7a0Os",
    "authDomain": "nmit-583b4.firebaseapp.com",
    "projectId": "nmit-583b4",
    "storageBucket": "nmit-583b4.appspot.com",
    "messagingSenderId": "86799504200",
    "appId": "1:86799504200:web:8a49cbe63c13770dfd197d",
    "measurementId": "G-KFE4EC438D",
    "serviceAccount": "serviceAccount.json",
    "databaseURL": "https://console.firebase.google.com/u/0/project/nmit-583b4/database/nmit-583b4-default-rtdb/data/~2F"
}

firebase=pyrebase.initialize_app(config)
storage=firebase.storage()
storage.child("Image11.jpg").put("Image11.jpg")