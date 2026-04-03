import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDN_jD5RCowH3LKHtOyKDjQm0cASWQqCRE",
  authDomain: "personal-ai-assistant-5a245.firebaseapp.com",
  projectId: "personal-ai-assistant-5a245",
  // storageBucket not needed — using free Spark plan (no Firebase Storage)
  messagingSenderId: "914225432637",
  appId: "1:914225432637:web:0f2961c1fae19991270254",
  measurementId: "G-0XQ5KK4826"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
// storage removed — using free Spark plan (no Firebase Storage)

export default app;