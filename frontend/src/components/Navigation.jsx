import React from 'react'
import { 
  CloudArrowUpIcon, 
  ChartBarIcon, 
  SparklesIcon, 
  ChartPieIcon,
  CheckIcon
} from '@heroicons/react/24/outline'
const Navigation = ({ currentSection, navigateToSection, wizardSteps }) => {
  const steps = [
    { id: 'ingestion', number: 1, title: 'Upload', subtitle: 'Data Files', icon: CloudArrowUpIcon },
    { id: 'quality', number: 2, title: 'Analyze', subtitle: 'Data Quality', icon: ChartBarIcon },
    { id: 'golden', number: 3, title: 'Merge', subtitle: 'Golden Record', icon: SparklesIcon },
    { id: 'analytics', number: 4, title: 'Insights', subtitle: 'Analytics', icon: ChartPieIcon }
  ]
  const getStepClass = (step) => {
    if (wizardSteps[step.id].completed) return 'wizard-step completed'
    if (currentSection === step.id) return 'wizard-step active'
    return 'wizard-step'
  }
  const getCurrentStepIndex = () => {
    return steps.findIndex(step => step.id === currentSection)
  }
  const progressPercentage = ((getCurrentStepIndex() + 1) / steps.length) * 100
  return (
    <nav className="sticky top-0 z-50 bg-gradient-to-r from-blue-600 via-blue-700 to-blue-800 text-white shadow-2xl backdrop-blur-lg border-b border-blue-500/20">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center relative z-10">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center backdrop-blur-sm">
                <ChartBarIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
                  Data Quality AI
                </h1>
                <span className="text-sm opacity-75 font-medium">Hackathon Junction</span>
              </div>
            </div>
          </div>
          {}
          <div className="wizard-container relative">
            <div 
              className="absolute top-0 left-0 h-1 bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 rounded-full transition-all duration-800 ease-out shadow-lg"
              style={{ width: `${progressPercentage}%` }}
            ></div>
            <div className="flex items-center justify-between space-x-8 relative z-10">
              {steps.map((step, index) => {
                const IconComponent = step.icon
                return (
                  <button
                    key={step.id}
                    onClick={() => navigateToSection(step.id)}
                    className={`${getStepClass(step)} group flex items-center px-6 py-4 text-sm font-medium transition-all duration-300 text-white cursor-pointer rounded-2xl hover:bg-white/10 backdrop-blur-sm`}
                  >
                    <div className="flex items-center space-x-4">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm relative overflow-hidden transition-all duration-300 shadow-lg ${
                        wizardSteps[step.id].completed 
                          ? 'bg-emerald-500 text-white' 
                          : currentSection === step.id 
                            ? 'bg-white text-blue-600' 
                            : 'bg-white/20 text-white group-hover:bg-white/30'
                      }`}>
                        {wizardSteps[step.id].completed ? (
                          <CheckIcon className="w-5 h-5" />
                        ) : (
                          <IconComponent className="w-5 h-5" />
                        )}
                      </div>
                      <div className="flex flex-col items-start">
                        <span className="font-bold text-white group-hover:text-blue-100 transition-colors">
                          {step.title}
                        </span>
                        <span className="text-xs text-white/80 group-hover:text-white/90 transition-colors">
                          {step.subtitle}
                        </span>
                      </div>
                    </div>
                    {index < steps.length - 1 && (
                      <div className="absolute top-1/2 -right-4 transform -translate-y-1/2 text-white/60 text-lg font-bold group-hover:text-white/80 transition-colors">
                        →
                      </div>
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
export default Navigation