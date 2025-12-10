// import { useState } from 'react'
import { StaticImageSegmentationPage } from "./components/StaticImageSegmentationPage";
import "./App.css";

function App() {
  return (
    <div className="app-root">
      <div className="app-container">
        <h1 className="app-title">Analyse d’Images avec IA</h1>
        <StaticImageSegmentationPage />
      </div>
    </div>
  );
}

export default App;

