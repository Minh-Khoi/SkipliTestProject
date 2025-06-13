// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getDatabase } from "firebase/database";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCwIZUhMLltV477le33DIIJLTtBKjaTJkc",
  authDomain: "anotherproject-28d3b.firebaseapp.com",
  databaseURL: "https://anotherproject-28d3b-default-rtdb.firebaseio.com",
  projectId: "anotherproject-28d3b",
  storageBucket: "anotherproject-28d3b.firebasestorage.app",
  messagingSenderId: "1005026079371",
  appId: "1:1005026079371:web:4e14de40e8367afcd19a8e"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const db = getDatabase(app);
