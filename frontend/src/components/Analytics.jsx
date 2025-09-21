import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { 
  ChartPieIcon, 
  ChartBarIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  LightBulbIcon,
  LinkIcon,
  Squares2X2Icon,
  DocumentTextIcon,
  ArrowPathIcon,
  PlayIcon
} from '@heroicons/react/24/outline'
const Analytics = ({ datasets, goldenRecord, apiBaseUrl }) => {
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState('')
  const availableDatasets = [...datasets]
  if (goldenRecord) {
    availableDatasets.push({
      dataset_id: 'golden_record',
      filename: 'Golden Record',
      rows: goldenRecord.total_records,
      columns: goldenRecord.total_columns
    })
  }
  useEffect(() => {
    if (availableDatasets.length > 0 && !selectedDataset) {
      setSelectedDataset(goldenRecord ? 'golden_record' : availableDatasets[0].dataset_id)
    }
  }, [availableDatasets, goldenRecord])
  const runAnalytics = async () => {
    if (!selectedDataset) return
    setLoading(true)
    try {
      const response = await axios.get(`${apiBaseUrl}/api/analytics/${selectedDataset}`)
      console.log('Advanced analytics response:', response.data)
      setAnalytics(response.data)
    } catch (error) {
      console.error('Analytics error:', error)
      alert(`Error running analytics: ${error.response?.data?.error || error.message}`)
    } finally {
      setLoading(false)
    }
  }
  if (availableDatasets.length === 0) {
    return (
      <div className="section">
        <div className="max-w-4xl mx-auto text-center">
          <div className="flex items-center space-x-3 mb-6">
            <ChartPieIcon className="w-8 h-8 text-blue-600" />
            <h2 className="text-3xl font-bold text-gray-900">Advanced Analytics</h2>
          </div>
          <div className="text-gray-500">
            <p className="text-lg mb-4">No datasets available for analysis</p>
            <p>Upload some data files or create a golden record to see analytics here.</p>
          </div>
        </div>
      </div>
    )
  }
  return (
    <div className="section">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center space-x-3 mb-6">
          <ChartPieIcon className="w-8 h-8 text-blue-600" />
          <h2 className="text-3xl font-bold text-gray-900">Advanced Analytics</h2>
        </div>
        {}
        <div className="bg-white rounded-xl shadow-lg border p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Dataset for Analysis</h3>
          <div className="flex flex-col sm:flex-row gap-4">
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {availableDatasets.map((dataset) => (
                <option key={dataset.dataset_id} value={dataset.dataset_id}>
                  {dataset.filename} ({dataset.rows?.toLocaleString()} rows, {dataset.columns} columns)
                </option>
              ))}
            </select>
            <button
              onClick={runAnalytics}
              disabled={loading}
              className="px-6 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <ArrowPathIcon className="w-4 h-4 animate-spin" />
                  <span>Analyzing...</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <PlayIcon className="w-4 h-4" />
                  <span>Run Advanced Analysis</span>
                </div>
              )}
            </button>
          </div>
        </div>
        {}
        {analytics && (
          <div className="space-y-6">
            {}
            {analytics.header && (
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <ChartBarIcon className="w-5 h-5 text-blue-600" />
                  <h4 className="text-xl font-semibold text-gray-900">Data Quality Assessment</h4>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <div className="text-2xl font-bold text-blue-600">
                      {analytics.header.credibility?.score || 0}
                    </div>
                    <div className="text-sm text-blue-800">Credibility Score</div>
                  </div>
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <div className="text-2xl font-bold text-indigo-600">
                      {analytics.header.credibility?.grade || 'F'}
                    </div>
                    <div className="text-sm text-indigo-800">Quality Grade</div>
                  </div>
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <div className="text-2xl font-bold text-green-600">
                      {analytics.header.rows?.toLocaleString() || 0}
                    </div>
                    <div className="text-sm text-green-800">Total Rows</div>
                  </div>
                  <div className="bg-white rounded-lg p-4 shadow-sm">
                    <div className="text-2xl font-bold text-purple-600">
                      {analytics.header.columns || 0}
                    </div>
                    <div className="text-sm text-purple-800">Total Columns</div>
                  </div>
                </div>
              </div>
            )}
            {}
            {analytics.body?.global_issues && analytics.body.global_issues.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <ExclamationTriangleIcon className="w-5 h-5 text-amber-600" />
                  <h4 className="text-xl font-semibold text-gray-900">Global Issues</h4>
                </div>
                <div className="space-y-3">
                  {analytics.body.global_issues.map((issue, idx) => (
                    <div key={idx} className="bg-red-50 rounded-lg p-4 border-l-4 border-red-400">
                      <div className="font-medium text-red-900">{issue}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {}
            {analytics.body?.column_reports && Object.keys(analytics.body.column_reports).length > 0 && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <DocumentTextIcon className="w-5 h-5 text-blue-600" />
                  <h4 className="text-xl font-semibold text-gray-900">Column Analysis</h4>
                </div>
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {Object.entries(analytics.body.column_reports).slice(0, 10).map(([colName, colData]) => (
                    <div key={colName} className="bg-gray-50 rounded-lg p-4">
                      <div className="font-medium text-gray-900 mb-2">{colData.description}</div>
                      {colData.issues && colData.issues.length > 0 && (
                        <div className="space-y-2">
                          {colData.issues.map((issue, idx) => (
                            <div key={idx} className="text-sm text-red-600 bg-red-100 rounded px-2 py-1">
                              <div className="flex items-center space-x-2">
                                <ExclamationTriangleIcon className="w-4 h-4 text-amber-500" />
                                <span>{issue}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      {colData.suggestions && colData.suggestions.length > 0 && (
                        <div className="space-y-1 mt-2">
                          {colData.suggestions.map((suggestion, idx) => (
                            <div key={idx} className={`text-xs rounded px-2 py-1 ${
                              suggestion.severity === 'major' ? 'bg-red-100 text-red-700' :
                              suggestion.severity === 'moderate' ? 'bg-yellow-100 text-yellow-700' :
                              'bg-blue-100 text-blue-700'
                            }`}>
                              <div className="flex items-center space-x-2">
                                <LightBulbIcon className="w-4 h-4 text-blue-500" />
                                <span>{suggestion.action}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {}
            {analytics.body?.correlations?.top_pairs && analytics.body.correlations.top_pairs.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <LinkIcon className="w-5 h-5 text-green-600" />
                  <h4 className="text-xl font-semibold text-gray-900">Strong Correlations</h4>
                </div>
                <div className="space-y-2">
                  {analytics.body.correlations.top_pairs.slice(0, 10).map((corr, idx) => (
                    <div key={idx} className="bg-gray-50 rounded-lg p-3">
                      <div className="font-medium text-gray-900">
                        {corr[0]} &lt;-&gt; {corr[1]}
                      </div>
                      <div className="text-sm text-gray-600">
                        Correlation: {corr[2].toFixed(3)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {}
            {analytics.body?.target_balance && Object.keys(analytics.body.target_balance).length > 0 && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <Squares2X2Icon className="w-5 h-5 text-purple-600" />
                  <h4 className="text-xl font-semibold text-gray-900">Target Distribution</h4>
                </div>
                <div className="space-y-2">
                  <div className="text-sm text-gray-600">
                    Column: {analytics.body.target_balance.column}
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    {Object.entries(analytics.body.target_balance.distribution_pct || {}).map(([key, value]) => (
                      <div key={key} className="bg-gray-50 rounded-lg p-3">
                        <div className="font-medium text-gray-900">Class {key}</div>
                        <div className="text-lg font-bold text-blue-600">{value.toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {}
            {analytics.body?.general_stats && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <ChartBarIcon className="w-5 h-5 text-indigo-600" />
                  <h4 className="text-xl font-semibold text-gray-900">Dataset Statistics</h4>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4">
                    <div className="text-lg font-bold text-blue-600">
                      {analytics.body.general_stats.duplicate_rows_pct?.toFixed(2) || 0}%
                    </div>
                    <div className="text-sm text-blue-800">Duplicate Rows</div>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4">
                    <div className="text-lg font-bold text-green-600">
                      {analytics.body.general_stats.empty_columns?.length || 0}
                    </div>
                    <div className="text-sm text-green-800">Empty Columns</div>
                  </div>
                  <div className="bg-orange-50 rounded-lg p-4">
                    <div className="text-lg font-bold text-orange-600">
                      {analytics.body.general_stats.constant_columns?.length || 0}
                    </div>
                    <div className="text-sm text-orange-800">Constant Columns</div>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-4">
                    <div className="text-lg font-bold text-purple-600">
                      {analytics.body.general_stats.candidate_id_columns?.length || 0}
                    </div>
                    <div className="text-sm text-purple-800">ID-like Columns</div>
                  </div>
                </div>
              </div>
            )}
            {}
            {analytics.footer && (
              <div className="bg-gray-50 rounded-xl p-4">
                <div className="text-sm text-gray-600 text-center">
                  {analytics.footer.notes?.join(' • ')} • Generated by {analytics.footer.generated_by}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
export default Analytics