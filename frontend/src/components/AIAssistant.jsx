import React, { useState, useEffect } from 'react'
import { 
  ChatBubbleLeftRightIcon, 
  XMarkIcon, 
  SparklesIcon,
  CloudArrowUpIcon,
  ChartBarIcon,
  SparklesIcon as GoldenIcon,
  ChartPieIcon,
  LightBulbIcon,
  WrenchScrewdriverIcon
} from '@heroicons/react/24/outline'
const AIAssistant = ({ currentSection }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([])
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        { text: "Welcome! I'm your AI Data Quality Assistant. I'm here to help you navigate through the wizard!", type: 'assistant', icon: 'welcome' },
        { text: "Click the buttons below for specific help, or just follow the steps: Upload → Analyze → Merge → Insights!", type: 'assistant', icon: 'steps' },
        { text: "Pro tip: I'll provide context-specific help as you progress through each step!", type: 'assistant', icon: 'tip' }
      ])
    }
  }, [])
  useEffect(() => {
    if (isOpen) {
      updateContextMessages()
    }
  }, [currentSection, isOpen])
  const updateContextMessages = () => {
    const contextMessages = {
      'ingestion': {
        message: "Ready to upload data? I can help you with file formats, encoding issues, and best practices!",
        icon: 'upload',
        tips: [
          "Tip: Your CSV files should have headers in the first row",
          "Pro tip: Files up to 400MB are supported with progress tracking",
          "Performance tip: CSV files process faster than Excel for large datasets"
        ]
      },
      'quality': {
        message: "Time to analyze data quality! I'll help you understand your quality metrics and AI suggestions.",
        icon: 'analyze',
        tips: [
          "Quality Score: 0-100 scale weighing multiple factors",
          "Missing values: Red metrics show data completeness issues",
          "AI Suggestions: Prioritized recommendations for data improvement"
        ]
      },
      'golden': {
        message: "Creating your golden record! I'll guide you through merging datasets and resolving conflicts.",
        icon: 'merge',
        tips: [
          "Merge Key: Choose a column that exists in all datasets",
          "Conflicts: I'll detect and resolve data inconsistencies",
          "Quality: The golden record will have an overall quality score"
        ]
      },
      'analytics': {
        message: "Advanced analytics time! I'll help you understand correlations, distributions, and anomalies.",
        icon: 'insights',
        tips: [
          "Correlations: Strong relationships between features",
          "Distributions: Statistical properties of your data",
          "Anomalies: ML-powered outlier detection using Isolation Forest"
        ]
      }
    }
    const context = contextMessages[currentSection]
    if (context && messages.length <= 3) {
      setMessages(prev => [
        ...prev.slice(0, 3),
        { text: context.message, type: 'assistant' },
        ...context.tips.map(tip => ({ text: tip, type: 'assistant' }))
      ])
    }
  }
  const showHelp = (helpType) => {
    const helpMessages = {
      'upload': [
        "📁 Upload Tips: CSV and Excel files are supported",
        "🔧 Encoding: I'll auto-detect file encoding issues",
        "📋 Headers: Make sure your first row contains column names",
        "💾 Size: Large files up to 400MB are handled efficiently"
      ],
      'quality': [
        "📊 Quality Analysis: I check missing values, duplicates, and data types",
        "📈 Statistics: Basic stats like mean, median, std are calculated",
        "🎯 Anomalies: Outliers are detected using statistical methods",
        "📋 Report: Each dataset gets a comprehensive quality score"
      ],
      'merge': [
        "💎 Golden Record: Merge multiple datasets on a common key",
        "🔗 Conflicts: I'll detect and resolve data conflicts",
        "🧹 Cleaning: Duplicates and inconsistencies are handled",
        "📊 Preview: See the merged result before downloading"
      ],
      'analytics': [
        "📈 Insights: Correlation analysis and distribution plots",
        "🤖 AI Detection: Advanced anomaly detection algorithms",
        "📊 Visualizations: Interactive charts and graphs",
        "💡 Recommendations: Data-driven suggestions for improvements"
      ]
    }
    const newMessages = helpMessages[helpType] || []
    setMessages(prev => [...prev, ...newMessages.map(text => ({ text, type: 'assistant' }))])
  }
  return (
    <div className="fixed bottom-5 right-5 z-50 max-w-sm">
      {!isOpen ? (
        <button
          onClick={() => setIsOpen(true)}
          className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110 flex items-center justify-center"
        >
          <ChatBubbleLeftRightIcon className="w-8 h-8" />
        </button>
      ) : (
        <div className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl shadow-2xl text-white max-w-md w-96">
          {}
          <div className="flex items-center justify-between p-4 border-b border-white border-opacity-20">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center mr-3">
                <SparklesIcon className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-bold text-lg">AI Assistant</h3>
                <p className="text-xs opacity-80">Data Quality Expert</p>
              </div>
            </div>
            <button 
              onClick={() => setIsOpen(false)}
              className="text-white hover:bg-white hover:bg-opacity-20 rounded-full p-1 transition-all"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>
          {}
          <div className="max-h-80 overflow-y-auto p-4 space-y-3">
            {messages.map((message, index) => (
              <div key={index} className="bg-white bg-opacity-20 rounded-lg p-3 text-sm">
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-white bg-opacity-30 rounded-full flex items-center justify-center mr-2 flex-shrink-0 mt-0.5">
                    <span className="text-xs">🤖</span>
                  </div>
                  <div className="flex-1">{message.text}</div>
                </div>
              </div>
            ))}
          </div>
          {}
          <div className="p-4 border-t border-white border-opacity-20">
            <div className="grid grid-cols-2 gap-2">
              <button 
                onClick={() => showHelp('upload')}
                className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-2 text-xs transition-all flex items-center space-x-1"
              >
                <CloudArrowUpIcon className="w-4 h-4" />
                <span>Upload Help</span>
              </button>
              <button 
                onClick={() => showHelp('quality')}
                className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-2 text-xs transition-all flex items-center space-x-1"
              >
                <ChartBarIcon className="w-4 h-4" />
                <span>Quality Tips</span>
              </button>
              <button 
                onClick={() => showHelp('merge')}
                className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-2 text-xs transition-all flex items-center space-x-1"
              >
                <GoldenIcon className="w-4 h-4" />
                <span>Merge Guide</span>
              </button>
              <button 
                onClick={() => showHelp('analytics')}
                className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg p-2 text-xs transition-all flex items-center space-x-1"
              >
                <ChartPieIcon className="w-4 h-4" />
                <span>Analytics</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
export default AIAssistant