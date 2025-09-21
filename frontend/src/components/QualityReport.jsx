import React, { useState, useEffect } from 'react'
import axios from 'axios'
import Chart from 'react-apexcharts'
import { 
  ChartBarIcon, 
  DocumentIcon, 
  SparklesIcon, 
  ArrowDownTrayIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  EyeIcon,
  FunnelIcon
} from '@heroicons/react/24/outline'
const QualityReport = ({ datasets, qualityReports, apiBaseUrl }) => {
  const [selectedDataset, setSelectedDataset] = useState('')
  const [comprehensiveReport, setComprehensiveReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [severityFilter, setSeverityFilter] = useState('all')
  const [showOnlyProblematic, setShowOnlyProblematic] = useState(false)
  const [cleaningInProgress, setCleaningInProgress] = useState(false)
  const [cleaningReport, setCleaningReport] = useState(null)
  const [cleanedDataset, setCleanedDataset] = useState(null)
  useEffect(() => {
    if (datasets.length > 0 && !selectedDataset) {
      setSelectedDataset(datasets[0].dataset_id)
    }
  }, [datasets, selectedDataset])
  useEffect(() => {
    if (selectedDataset) {
      loadComprehensiveReport(selectedDataset)
    }
  }, [selectedDataset])
  const loadComprehensiveReport = async (datasetId) => {
    setLoading(true)
    try {
      const response = await axios.get(`${apiBaseUrl}/api/comprehensive-report/${datasetId}`)
      console.log('Comprehensive report response:', response.data)
      setComprehensiveReport(response.data)
    } catch (error) {
      console.error('Error loading comprehensive report:', error)
      setComprehensiveReport(null)
    } finally {
      setLoading(false)
    }
  }
  const applyDataCleaning = async () => {
    if (!selectedDataset) return
    setCleaningInProgress(true)
    try {
      console.log('🧹 Starting data cleaning...')
      const response = await axios.post(`${apiBaseUrl}/api/clean-dataset/${selectedDataset}`)
      console.log('Cleaning response:', response.data)
      setCleaningReport(response.data.cleaning_report)
      setCleanedDataset(response.data)
      alert(`✅ Data cleaning completed!\n\n📊 Original: ${response.data.original_shape[0]} rows × ${response.data.original_shape[1]} columns\n📊 Cleaned: ${response.data.cleaned_shape[0]} rows × ${response.data.cleaned_shape[1]} columns\n📈 Quality score improved: ${response.data.cleaning_report.quality_improvement.original_score} → ${response.data.cleaning_report.quality_improvement.cleaned_score || 'N/A'}`)
    } catch (error) {
      console.error('Error cleaning dataset:', error)
      alert(`❌ Error cleaning dataset: ${error.response?.data?.error || error.message}`)
    } finally {
      setCleaningInProgress(false)
    }
  }
  const downloadCleanedDataset = () => {
    if (cleanedDataset?.download_url) {
      window.open(`${apiBaseUrl}${cleanedDataset.download_url}`, '_blank')
    }
  }
  const filterColumns = (columnReports) => {
    if (!columnReports) return {}
    if (!showOnlyProblematic) return columnReports
    const filtered = {}
    Object.entries(columnReports).forEach(([colName, report]) => {
      const hasIssues = report.issues && report.issues.length > 0
      const hasSuggestions = report.suggestions && report.suggestions.length > 0
      if (hasIssues || hasSuggestions) {
        if (hasSuggestions) {
          const hasMatchingSeverity = report.suggestions.some(sugg => 
            sugg.severity === severityFilter || 
            (severityFilter === 'all' && sugg.severity)
          )
          if (hasMatchingSeverity) {
            filtered[colName] = report
          }
        } else if (hasIssues) {
          filtered[colName] = report
        }
      }
    })
    return filtered
  }
  if (datasets.length === 0) {
    return (
      <div className="section">
        <div className="max-w-4xl mx-auto text-center">
          <div className="flex items-center space-x-3 mb-6">
            <ChartBarIcon className="w-8 h-8 text-blue-600" />
            <h2 className="text-3xl font-bold text-gray-900">Quality Report</h2>
          </div>
          <div className="text-gray-500">
            <p className="text-lg mb-4">No datasets uploaded yet</p>
            <p>Upload some data files in the Data Ingestion section to see quality reports here.</p>
          </div>
        </div>
      </div>
    )
  }
  const filteredColumns = comprehensiveReport ? filterColumns(comprehensiveReport.body.column_reports) : {}
  return (
    <div className="section">
      <div className="max-w-full mx-auto">
        <div className="flex items-center justify-center space-x-3 mb-8">
          <ChartBarIcon className="w-8 h-8 text-blue-600" />
          <h2 className="text-3xl font-bold text-gray-900">Comprehensive Data Quality Reports</h2>
        </div>
        <div className="flex h-screen">
          {}
          <div className="w-80 bg-white border-r border-gray-200 shadow-lg">
            <div className="p-6">
              <div className="flex items-center space-x-2 mb-4">
                <DocumentIcon className="w-5 h-5 text-gray-700" />
                <h3 className="text-lg font-semibold text-gray-900">Datasets</h3>
              </div>
              <div className="space-y-2">
                {datasets.map((dataset) => (
                  <button
                    key={dataset.dataset_id}
                    onClick={() => setSelectedDataset(dataset.dataset_id)}
                    className={`w-full text-left p-4 rounded-lg transition-all duration-300 transform hover:scale-105 ${
                      selectedDataset === dataset.dataset_id
                        ? 'bg-blue-600 text-white shadow-lg ring-2 ring-blue-300'
                        : 'bg-gray-50 text-gray-700 border border-gray-200 hover:bg-blue-50 hover:border-blue-300'
                    }`}
                  >
                    <div className="font-bold text-sm mb-1 truncate" title={dataset.filename}>
                      {dataset.filename}
                    </div>
                    <div className="text-xs opacity-75">
                      {dataset.rows.toLocaleString()} rows × {dataset.columns} cols
                    </div>
                    <div className="text-xs opacity-60 mt-1">
                      {(dataset.file_size / (1024 * 1024)).toFixed(1)} MB
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
          {}
          <div className="flex-1 overflow-y-auto">
            {loading ? (
              <div className="flex justify-center items-center h-full">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Loading comprehensive report...</p>
                </div>
              </div>
            ) : comprehensiveReport ? (
              <div className="p-8 space-y-8">
                {}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl shadow-inner border border-blue-200">
                  <h3 className="text-2xl font-bold text-blue-800 mb-4">
                    {comprehensiveReport.header.dataset_name}
                  </h3>
                  {}
                  {comprehensiveReport.header.description && (
                    <div className="mb-4 p-4 bg-white rounded-lg border border-blue-200">
                      <p className="text-gray-700 text-sm leading-relaxed">
                        {comprehensiveReport.header.description}
                      </p>
                    </div>
                  )}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-gray-700">
                    <div>
                      <span className="font-semibold">Rows:</span> {comprehensiveReport.header.rows.toLocaleString()}
                    </div>
                    <div>
                      <span className="font-semibold">Columns:</span> {comprehensiveReport.header.columns}
                    </div>
                    <div>
                      <span className="font-semibold">Credibility Score:</span>{' '}
                      <span className={`font-bold text-lg ${
                        comprehensiveReport.header.credibility.grade === 'A' ? 'text-green-600' :
                        comprehensiveReport.header.credibility.grade === 'B' ? 'text-lime-600' :
                        comprehensiveReport.header.credibility.grade === 'C' ? 'text-yellow-600' :
                        comprehensiveReport.header.credibility.grade === 'D' ? 'text-orange-600' : 'text-red-600'
                      }`}>
                        {comprehensiveReport.header.credibility.score} ({comprehensiveReport.header.credibility.grade})
                      </span>
                    </div>
                    <div>
                      <span className="font-semibold">Status:</span>{' '}
                      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                        comprehensiveReport.header.credibility.grade === 'A' ? 'bg-green-100 text-green-800' :
                        comprehensiveReport.header.credibility.grade === 'B' ? 'bg-lime-100 text-lime-800' :
                        comprehensiveReport.header.credibility.grade === 'C' ? 'bg-yellow-100 text-yellow-800' :
                        comprehensiveReport.header.credibility.grade === 'D' ? 'bg-orange-100 text-orange-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {comprehensiveReport.header.credibility.grade} Grade
                      </span>
                    </div>
                  </div>
                  {}
                  <div className="mt-6 flex flex-wrap gap-4">
                    <button
                      onClick={applyDataCleaning}
                      disabled={cleaningInProgress}
                      className={`px-6 py-3 rounded-lg font-semibold text-white transition-all duration-300 transform hover:scale-105 ${
                        cleaningInProgress
                          ? 'bg-gray-400 cursor-not-allowed'
                          : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 shadow-lg hover:shadow-xl'
                      }`}
                    >
                      {cleaningInProgress ? (
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                          <span>Cleaning Data...</span>
                        </div>
                      ) : (
                        <div className="flex items-center space-x-2">
                          <span>🧹</span>
                          <span>Apply Suggestions & Clean Data</span>
                        </div>
                      )}
                    </button>
                    {cleanedDataset && (
                      <button
                        onClick={downloadCleanedDataset}
                        className="px-6 py-3 rounded-lg font-semibold text-white bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
                      >
                        <div className="flex items-center space-x-2">
                          <span>📥</span>
                          <span>Download Cleaned Dataset</span>
                        </div>
                      </button>
                    )}
                  </div>
                </div>
                {}
                {cleaningReport && (
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-6 rounded-xl shadow-inner border border-green-200">
                    <h4 className="text-xl font-bold text-green-800 mb-4 flex items-center">
                      <span className="mr-2">✅</span>
                      Data Cleaning Report
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                      <div className="bg-white p-4 rounded-lg border border-green-200">
                        <div className="text-sm text-gray-600">Original Size</div>
                        <div className="text-lg font-bold text-gray-900">
                          {cleaningReport.cleaning_summary?.original_shape?.[0]?.toLocaleString() || 'N/A'} × {cleaningReport.cleaning_summary?.original_shape?.[1] || 'N/A'}
                        </div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border border-green-200">
                        <div className="text-sm text-gray-600">Cleaned Size</div>
                        <div className="text-lg font-bold text-gray-900">
                          {cleaningReport.cleaning_summary?.cleaned_shape?.[0]?.toLocaleString() || 'N/A'} × {cleaningReport.cleaning_summary?.cleaned_shape?.[1] || 'N/A'}
                        </div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border border-green-200">
                        <div className="text-sm text-gray-600">Quality Score</div>
                        <div className="text-lg font-bold text-green-600">
                          {cleaningReport.quality_improvement.original_score} → {cleaningReport.quality_improvement.cleaned_score || 'N/A'}
                        </div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border border-green-200">
                        <div className="text-sm text-gray-600">Memory Saved</div>
                        <div className="text-lg font-bold text-blue-600">
                          {cleaningReport.cleaning_summary?.memory_reduction_mb || 0} MB
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <h5 className="font-semibold text-green-800 mb-2">Transformations Applied:</h5>
                        <ul className="space-y-1 text-sm text-gray-700">
                          {(cleaningReport.transformations || []).map((transformation, index) => (
                            <li key={index} className="flex items-start">
                              <span className="text-green-500 mr-2">•</span>
                              <span>{transformation}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      {cleaningReport.recommendations && cleaningReport.recommendations.length > 0 && (
                        <div>
                          <h5 className="font-semibold text-blue-800 mb-2">Recommendations:</h5>
                          <ul className="space-y-1 text-sm text-gray-700">
                            {(cleaningReport.recommendations || []).map((recommendation, index) => (
                              <li key={index} className="flex items-start">
                                <span className="text-blue-500 mr-2">💡</span>
                                <span>{recommendation}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                {}
                {comprehensiveReport.body.charts && Object.keys(comprehensiveReport.body.charts).length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg border p-6">
                    <h4 className="text-xl font-semibold text-gray-900 mb-6">📈 Data Visualizations</h4>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      {}
                      {comprehensiveReport.body.charts.missingness && (
                        <div>
                          <Chart
                            options={comprehensiveReport.body.charts.missingness.options}
                            series={comprehensiveReport.body.charts.missingness.series}
                            type="bar"
                            height={400}
                          />
                        </div>
                      )}
                      {}
                      {comprehensiveReport.body.charts.outliers && (
                        <div>
                          <Chart
                            options={comprehensiveReport.body.charts.outliers.options}
                            series={comprehensiveReport.body.charts.outliers.series}
                            type="bar"
                            height={400}
                          />
                        </div>
                      )}
                      {}
                      {comprehensiveReport.body.charts.correlations && (
                        <div>
                          <Chart
                            options={comprehensiveReport.body.charts.correlations.options}
                            series={comprehensiveReport.body.charts.correlations.series}
                            type="bar"
                            height={400}
                          />
                        </div>
                      )}
                      {}
                      {comprehensiveReport.body.charts.target_balance && (
                        <div>
                          <Chart
                            options={comprehensiveReport.body.charts.target_balance.options}
                            series={comprehensiveReport.body.charts.target_balance.series}
                            type="pie"
                            height={400}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}
                {}
                {comprehensiveReport.body.global_issues && comprehensiveReport.body.global_issues.length > 0 && (
                  <div className="bg-red-50 border border-red-200 rounded-xl shadow-sm p-6">
                    <h4 className="text-xl font-semibold text-red-800 mb-4 flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      Global Data Quality Issues
                    </h4>
                    <ul className="list-disc list-inside text-red-700 space-y-2">
                      {(comprehensiveReport.body?.global_issues || []).map((issue, idx) => (
                        <li key={idx}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {}
                <div className="bg-white rounded-xl shadow-lg border p-6">
                  <h4 className="text-xl font-semibold text-gray-900 mb-4">📊 General Statistics</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-gray-700">
                    <div>
                      <span className="font-semibold">Duplicate Rows:</span> {comprehensiveReport.body.general_stats.duplicate_rows_pct}%
                    </div>
                    <div>
                      <span className="font-semibold">Empty Columns:</span> {comprehensiveReport.body.general_stats.empty_columns.join(', ') || 'None'}
                    </div>
                    <div>
                      <span className="font-semibold">Constant Columns:</span> {comprehensiveReport.body.general_stats.constant_columns.join(', ') || 'None'}
                    </div>
                    <div>
                      <span className="font-semibold">Candidate ID Columns:</span> {comprehensiveReport.body.general_stats.candidate_id_columns.join(', ') || 'None'}
                    </div>
                  </div>
                </div>
                {}
                <div className="bg-white rounded-xl shadow-lg border p-6">
                  <h4 className="text-xl font-semibold text-gray-900 mb-4">🔍 Column Filter</h4>
                  <div className="flex flex-wrap gap-4 items-center">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="showOnlyProblematic"
                        checked={showOnlyProblematic}
                        onChange={(e) => setShowOnlyProblematic(e.target.checked)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <label htmlFor="showOnlyProblematic" className="text-sm font-medium text-gray-700">
                        Show only columns with issues
                      </label>
                    </div>
                    {showOnlyProblematic && (
                      <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium text-gray-700">Severity:</label>
                        <select
                          value={severityFilter}
                          onChange={(e) => setSeverityFilter(e.target.value)}
                          className="rounded border-gray-300 text-sm focus:ring-blue-500 focus:border-blue-500"
                        >
                          <option value="all">All</option>
                          <option value="major">Major</option>
                          <option value="moderate">Moderate</option>
                          <option value="minor">Minor</option>
                        </select>
                      </div>
                    )}
                    <div className="text-sm text-gray-500">
                      Showing {Object.keys(filteredColumns).length} of {Object.keys(comprehensiveReport.body.column_reports || {}).length} columns
                    </div>
                  </div>
                </div>
                {}
                <div className="bg-white rounded-xl shadow-lg border p-6">
                  <h4 className="text-xl font-semibold text-gray-900 mb-4">
                    Detailed Column Reports {showOnlyProblematic && `(${Object.keys(filteredColumns).length} with issues)`}
                  </h4>
                  <div className="space-y-6">
                    {Object.keys(filteredColumns).length > 0 ? (
                      Object.entries(filteredColumns).map(([colName, report]) => (
                        <div key={colName} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                          <h5 className="text-lg font-semibold text-gray-800 mb-2">{colName}</h5>
                          <p className="text-gray-600 text-sm mb-3">{report.description}</p>
                          {report.issues && report.issues.length > 0 && (
                            <div className="mb-3">
                              <h6 className="font-medium text-red-700 text-sm mb-1">Issues:</h6>
                              <ul className="list-disc list-inside text-red-600 text-sm space-y-1">
                                {(report.issues || []).map((issue, idx) => <li key={idx}>{issue}</li>)}
                              </ul>
                            </div>
                          )}
                          {report.suggestions && report.suggestions.length > 0 && (
                            <div>
                              <h6 className="font-medium text-blue-700 text-sm mb-1">Suggestions:</h6>
                              <ul className="list-disc list-inside text-blue-600 text-sm space-y-1">
                                {(report.suggestions || []).map((sugg, idx) => (
                                  <li key={idx}>
                                    <span className={`font-semibold ${
                                      sugg.severity === 'major' ? 'text-red-700' :
                                      sugg.severity === 'moderate' ? 'text-orange-700' : 'text-green-700'
                                    }`}>
                                      [{sugg.severity.charAt(0).toUpperCase() + sugg.severity.slice(1)}]
                                    </span>{' '}
                                    {sugg.action}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <p>No columns match the current filter criteria.</p>
                        <p className="text-sm mt-2">Try adjusting the severity filter or uncheck "Show only columns with issues".</p>
                      </div>
                    )}
                  </div>
                </div>
                {}
                {comprehensiveReport.footer?.notes && comprehensiveReport.footer.notes.length > 0 && (
                  <div className="text-center text-gray-500 text-sm mt-8">
                    {(comprehensiveReport.footer?.notes || []).map((note, idx) => <p key={idx}>{note}</p>)}
                    <p>Generated by: {comprehensiveReport.footer.generated_by}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex justify-center items-center h-full">
                <div className="text-center">
                  <div className="text-gray-500">
                    <p className="text-lg mb-4">No comprehensive report available</p>
                    <p>Click on a dataset tab to load its detailed quality report.</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
export default QualityReport