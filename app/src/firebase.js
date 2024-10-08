// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDS0xa1m9cdu0SQzX_GkaoN7UMITPwPQbg",
  authDomain: "spanos-eds.firebaseapp.com",
  projectId: "spanos-eds",
  storageBucket: "spanos-eds.appspot.com",
  messagingSenderId: "131238705669",
  appId: "1:131238705669:web:82cb26b3541a07ff913a6c",
  measurementId: "G-JT4B4ZS4K5"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);