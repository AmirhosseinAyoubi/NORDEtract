import React, { useState, useEffect } from 'react'
import Navigation from './components/Navigation'
import DataIngestion from './components/DataIngestion'
import QualityReport from './components/QualityReport'
import GoldenRecord from './components/GoldenRecord'
import Analytics from './components/Analytics'
import AIAssistant from './components/AIAssistant'
const API_BASE_URL = 'http://localhost:5000'
function App() {
  const [currentSection, setCurrentSection] = useState('ingestion')
  const [datasets, setDatasets] = useState([])
  const [qualityReports, setQualityReports] = useState({})
  const [goldenRecord, setGoldenRecord] = useState(null)
  const [wizardSteps, setWizardSteps] = useState({
    ingestion: { unlocked: true, completed: false },
    quality: { unlocked: true, completed: false },
    golden: { unlocked: true, completed: false },
    analytics: { unlocked: true, completed: false }
  })
  useEffect(() => {
    setWizardSteps(prev => ({
      ingestion: { unlocked: true, completed: datasets.length > 0 },
      quality: { unlocked: true, completed: datasets.length > 0 && Object.keys(qualityReports).length > 0 },
      golden: { unlocked: true, completed: goldenRecord !== null },
      analytics: { unlocked: true, completed: false }
    }))
  }, [datasets, qualityReports, goldenRecord])
  const navigateToSection = (sectionName) => {
    setCurrentSection(sectionName)
  }
  const addDataset = (dataset) => {
    setDatasets(prev => [...prev, dataset])
    if (dataset.quality_report) {
      setQualityReports(prev => ({
        ...prev,
        [dataset.dataset_id]: dataset.quality_report
      }))
    }
  }
  const setGoldenRecordData = (record) => {
    setGoldenRecord(record)
  }
  return (
    <div className="min-h-screen">
      <Navigation 
        currentSection={currentSection}
        navigateToSection={navigateToSection}
        wizardSteps={wizardSteps}
      />
      <main>
        {currentSection === 'ingestion' && (
          <DataIngestion 
            datasets={datasets}
            addDataset={addDataset}
            apiBaseUrl={API_BASE_URL}
          />
        )}
        {currentSection === 'quality' && (
          <QualityReport 
            datasets={datasets}
            qualityReports={qualityReports}
            apiBaseUrl={API_BASE_URL}
          />
        )}
        {currentSection === 'golden' && (
          <GoldenRecord 
            datasets={datasets}
            goldenRecord={goldenRecord}
            setGoldenRecord={setGoldenRecordData}
            apiBaseUrl={API_BASE_URL}
          />
        )}
        {currentSection === 'analytics' && (
          <Analytics 
            datasets={datasets}
            goldenRecord={goldenRecord}
            apiBaseUrl={API_BASE_URL}
          />
        )}
      </main>
      <AIAssistant currentSection={currentSection} />
    </div>
  )
}
export default App