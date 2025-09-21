import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { 
  SparklesIcon, 
  CheckCircleIcon,
  ArrowPathIcon,
  MagnifyingGlassIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
  KeyIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'
const GoldenRecord = ({ datasets, goldenRecord, setGoldenRecord, apiBaseUrl }) => {
  const [loading, setLoading] = useState(false)
  const [mergeKey, setMergeKey] = useState('')
  const [commonKeys, setCommonKeys] = useState([])
  const [keysLoading, setKeysLoading] = useState(false)
  const [selectedKeyInfo, setSelectedKeyInfo] = useState(null)
  useEffect(() => {
    if (datasets.length >= 2) {
      loadCommonKeys()
    }
  }, [datasets])
  const loadCommonKeys = async () => {
    setKeysLoading(true)
    try {
      const response = await axios.post(`${apiBaseUrl}/api/common-keys`, {
        datasets: datasets
      })
      setCommonKeys(response.data.common_keys || [])
      if (response.data.common_keys && response.data.common_keys.length > 0) {
        const bestKey = response.data.common_keys[0]
        setMergeKey(bestKey.column)
        setSelectedKeyInfo(bestKey)
      }
    } catch (error) {
      console.error('Error loading common keys:', error)
    } finally {
      setKeysLoading(false)
    }
  }
  const handleKeySelection = (key) => {
    setMergeKey(key.column)
    setSelectedKeyInfo(key)
  }
  if (datasets.length < 2) {
    return (
      <div className="section">
        <div className="max-w-4xl mx-auto text-center">
          <div className="flex items-center space-x-3 mb-6">
            <SparklesIcon className="w-8 h-8 text-purple-600" />
            <h2 className="text-3xl font-bold text-gray-900">Golden Record</h2>
          </div>
          <div className="text-gray-500">
            <p className="text-lg mb-4">Need at least 2 datasets to create a golden record</p>
            <p>Upload multiple data files in the Data Ingestion section to merge them here.</p>
          </div>
        </div>
      </div>
    )
  }
  const createGoldenRecord = async () => {
    if (!mergeKey) {
      alert('Please select a merge key first.')
      return
    }
    setLoading(true)
    try {
      const response = await axios.post(`${apiBaseUrl}/api/merge`, {
        merge_key: mergeKey,
        datasets: datasets
      })
      setGoldenRecord(response.data)
      alert('Golden record created successfully!')
    } catch (error) {
      console.error('Golden record creation error:', error)
      if (error.response?.data?.error) {
        alert(`Error creating golden record: ${error.response.data.error}`)
      } else {
        alert(`Error creating golden record: ${error.message}`)
      }
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="section">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center space-x-3 mb-6">
          <SparklesIcon className="w-8 h-8 text-purple-600" />
          <h2 className="text-3xl font-bold text-gray-900">Golden Record Creation</h2>
        </div>
        {!goldenRecord ? (
          <div className="space-y-6">
            {}
            <div className="bg-white rounded-xl shadow-lg border p-6">
              <div className="flex items-center space-x-2 mb-4">
                <KeyIcon className="w-5 h-5 text-blue-600" />
                <h3 className="text-lg font-semibold text-gray-900">Detected Common Keys</h3>
                {keysLoading && <ArrowPathIcon className="w-4 h-4 animate-spin text-blue-600" />}
              </div>
              {keysLoading ? (
                <div className="text-center py-8">
                  <ArrowPathIcon className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-2" />
                  <p className="text-gray-600">Analyzing datasets for common keys...</p>
                </div>
              ) : commonKeys.length > 0 ? (
                <div className="space-y-4">
                  <p className="text-sm text-gray-600 mb-4">
                    Found {commonKeys.length} common column{commonKeys.length !== 1 ? 's' : ''} across all datasets. 
                    Select the best one for merging:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {commonKeys.map((key, index) => (
                      <div 
                        key={key.column}
                        className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
                          mergeKey === key.column 
                            ? 'border-blue-500 bg-blue-50' 
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => handleKeySelection(key)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium text-gray-900">{key.column}</h4>
                          <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                            key.merge_suitability === 'excellent' ? 'bg-green-100 text-green-800' :
                            key.merge_suitability === 'good' ? 'bg-blue-100 text-blue-800' :
                            key.merge_suitability === 'fair' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {key.merge_suitability}
                          </div>
                        </div>
                        <div className="text-sm text-gray-600 space-y-1">
                          <div>Total unique values: {key.total_unique_values.toLocaleString()}</div>
                          <div className="flex items-center space-x-4">
                            <span>Datasets: {Object.keys(key.datasets).length}</span>
                            <span>Avg nulls: {(
                              Object.values(key.datasets).reduce((sum, info) => sum + info.null_percentage, 0) / 
                              Object.keys(key.datasets).length
                            ).toFixed(1)}%</span>
                          </div>
                        </div>
                        {mergeKey === key.column && (
                          <div className="mt-3 pt-3 border-t border-gray-200">
                            <div className="text-xs text-gray-500">
                              <div className="font-medium mb-1">Sample values:</div>
                              <div className="flex flex-wrap gap-1">
                                {Object.values(key.datasets)[0]?.sample_values?.slice(0, 3).map((val, idx) => (
                                  <span key={idx} className="bg-gray-100 px-2 py-1 rounded text-xs">
                                    {String(val).substring(0, 20)}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <ExclamationTriangleIcon className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
                  <h4 className="text-lg font-medium text-gray-900 mb-2">No Common Keys Found</h4>
                  <p className="text-gray-600">
                    No columns are common across all datasets. You may need to upload datasets with shared identifiers.
                  </p>
                </div>
              )}
            </div>
            {}
            {commonKeys.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Merge Configuration</h3>
                {selectedKeyInfo && (
                  <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <InformationCircleIcon className="w-5 h-5 text-blue-600" />
                      <span className="font-medium text-blue-900">Selected Key: {selectedKeyInfo.column}</span>
                    </div>
                    <div className="text-sm text-blue-800">
                      This key will be used to join all {Object.keys(selectedKeyInfo.datasets).length} datasets
                    </div>
                  </div>
                )}
                {}
                <div className="mb-6">
                  <h4 className="font-medium text-gray-900 mb-3">Datasets to Merge</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {datasets.map((dataset, index) => (
                      <div key={index} className="bg-gray-50 rounded-lg p-4">
                        <div className="font-medium text-gray-900">{dataset.filename}</div>
                        <div className="text-sm text-gray-500">
                          {dataset.rows.toLocaleString()} rows × {dataset.columns} columns
                        </div>
                        <div className="mt-2 text-xs text-green-600 flex items-center space-x-1">
                          <CheckCircleIcon className="w-3 h-3" />
                          <span>Ready for merging</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <button 
                  onClick={createGoldenRecord}
                  disabled={loading || !mergeKey}
                  className="w-full premium-button py-3 px-6 text-lg disabled:opacity-50"
                >
                  {loading ? (
                    <div className="flex items-center space-x-2">
                      <ArrowPathIcon className="w-4 h-4 animate-spin" />
                      <span>Creating Golden Record...</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-2">
                      <SparklesIcon className="w-4 h-4" />
                      <span>Create Golden Record</span>
                    </div>
                  )}
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <div className="flex items-center space-x-2">
                    <CheckCircleIcon className="w-6 h-6 text-green-600" />
                    <h3 className="text-xl font-bold text-green-900">Golden Record Created</h3>
                  </div>
                  <p className="text-green-700">Successfully merged {datasets.length} datasets</p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-green-600">
                    {goldenRecord.quality_score?.toFixed(1)}%
                  </div>
                  <div className="text-sm text-green-700">Quality Score</div>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-600">{goldenRecord.quality_summary?.rows?.toLocaleString() || 0}</div>
                  <div className="text-blue-800 font-medium">Total Records</div>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <div className="text-2xl font-bold text-purple-600">{goldenRecord.quality_summary?.columns || 0}</div>
                  <div className="text-purple-800 font-medium">Total Columns</div>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <div className="text-2xl font-bold text-orange-600">{goldenRecord.quality_summary?.features_created || 0}</div>
                  <div className="text-orange-800 font-medium">Features Created</div>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-600">{goldenRecord.quality_summary?.completeness?.toFixed(1) || 0}%</div>
                  <div className="text-green-800 font-medium">Data Completeness</div>
                </div>
              </div>
            </div>
            {}
            {goldenRecord.merge_info && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <SparklesIcon className="w-5 h-5 text-purple-600" />
                  <h4 className="text-lg font-semibold text-gray-900">Advanced Merge Features</h4>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-medium text-gray-900 mb-2">Feature Engineering</h5>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Features Created:</span>
                        <span className="font-medium">{goldenRecord.merge_info?.feature_engineering?.features_created || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Merge Time:</span>
                        <span className="font-medium">{goldenRecord.merge_info?.merge_summary?.merge_time_seconds || 0}s</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Datasets Merged:</span>
                        <span className="font-medium">{goldenRecord.merge_info?.merge_summary?.datasets_merged || 0}</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-medium text-gray-900 mb-2">Data Quality</h5>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Completeness:</span>
                        <span className="font-medium">{goldenRecord.merge_info?.data_quality?.completeness || 0}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Memory Usage:</span>
                        <span className="font-medium">{goldenRecord.merge_info?.data_quality?.memory_usage_mb || 0} MB</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Duplicate Rows:</span>
                        <span className="font-medium">{goldenRecord.merge_info?.data_quality?.duplicate_rows || 0}</span>
                      </div>
                    </div>
                  </div>
                </div>
                {goldenRecord.merge_info?.feature_engineering?.feature_list && (
                  <div className="mt-4">
                    <h5 className="font-medium text-gray-900 mb-2">Sample Features Created:</h5>
                    <div className="flex flex-wrap gap-2">
                      {goldenRecord.merge_info.feature_engineering.feature_list.slice(0, 8).map((feature, idx) => (
                        <span key={idx} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                          {feature}
                        </span>
                      ))}
                      {goldenRecord.merge_info.feature_engineering.feature_list.length > 8 && (
                        <span className="text-gray-500 text-xs">
                          +{goldenRecord.merge_info.feature_engineering.feature_list.length - 8} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
            {}
            {goldenRecord.conflicts && goldenRecord.conflicts.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <MagnifyingGlassIcon className="w-5 h-5 text-blue-600" />
                  <h4 className="text-lg font-semibold text-gray-900">Resolved Conflicts</h4>
                </div>
                <div className="space-y-3">
                  {goldenRecord.conflicts.slice(0, 10).map((conflict, idx) => (
                    <div key={idx} className="bg-orange-50 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-medium">Key: {conflict.key}</span>
                          <span className="mx-2">•</span>
                          <span className="text-gray-600">Column: {conflict.column}</span>
                        </div>
                        <span className="text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded">
                          {conflict.resolution}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        <span className="line-through">{conflict.new_value}</span> → <span className="font-medium">{conflict.existing_value}</span>
                      </div>
                    </div>
                  ))}
                  {goldenRecord.conflicts.length > 10 && (
                    <div className="text-sm text-gray-500 text-center">
                      ... and {goldenRecord.conflicts.length - 10} more conflicts resolved
                    </div>
                  )}
                </div>
              </div>
            )}
            {}
            {goldenRecord.preview && (
              <div className="bg-white rounded-xl shadow-lg border p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <DocumentTextIcon className="w-5 h-5 text-gray-600" />
                  <h4 className="text-lg font-semibold text-gray-900">Golden Record Preview</h4>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white border rounded-lg">
                    <thead className="bg-gray-50">
                      <tr>
                        {Object.keys(goldenRecord.preview[0] || {}).map((column, idx) => (
                          <th key={idx} className="px-4 py-2 text-left text-sm font-medium text-gray-700 border-b">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {goldenRecord.preview.slice(0, 5).map((row, idx) => (
                        <tr key={idx} className="hover:bg-gray-50">
                          {Object.values(row).map((value, valueIdx) => (
                            <td key={valueIdx} className="px-4 py-2 text-sm text-gray-600 border-b">
                              {value === null || value === undefined ? (
                                <span className="text-red-500 italic">null</span>
                              ) : (
                                String(value)
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            <div className="text-center">
              <button 
                onClick={() => setGoldenRecord(null)}
                className="premium-button"
              >
                🔄 Create New Golden Record
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
export default GoldenRecord