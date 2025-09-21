import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { 
  CloudArrowUpIcon, 
  DocumentIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon,
  XCircleIcon,
  ArrowDownTrayIcon,
  TrashIcon,
  EyeIcon
} from '@heroicons/react/24/outline'
const DataIngestion = ({ datasets, addDataset, apiBaseUrl }) => {
  const [uploadQueue, setUploadQueue] = useState([])
  const [activeUploads, setActiveUploads] = useState(new Map())
  const uploadFile = async (file) => {
    const fileId = `${file.name}_${Date.now()}`
    const fileSizeMB = file.size / (1024 * 1024)
    setActiveUploads(prev => new Map(prev.set(fileId, {
      file,
      progress: 0,
      status: 'Starting upload...',
      startTime: Date.now()
    })))
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await axios.post(`${apiBaseUrl}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 1200000,
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        responseType: 'json',
        transformResponse: [(data) => {
          if (typeof data === 'string') {
            try {
              return JSON.parse(data)
            } catch (e) {
              console.error('JSON parse error:', e)
              return data
            }
          }
          return data
        }],
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setActiveUploads(prev => {
            const newMap = new Map(prev)
            const uploadInfo = newMap.get(fileId)
            if (uploadInfo) {
              newMap.set(fileId, {
                ...uploadInfo,
                progress: percentCompleted,
                status: percentCompleted === 100 
                  ? `Processing ${fileSizeMB.toFixed(1)}MB file on server...` 
                  : `Uploading ${fileSizeMB.toFixed(1)}MB... ${percentCompleted}%`
              })
            }
            return newMap
          })
        },
      })
      console.log('Upload response:', response.data)
      console.log('Response status:', response.status)
      console.log('Response headers:', response.headers)
      console.log('Response type:', typeof response.data)
      let responseData = response.data
      if (typeof responseData === 'string') {
        try {
          responseData = JSON.parse(responseData)
          console.log('Parsed response data:', responseData)
        } catch (e) {
          console.error('Failed to parse response as JSON:', e)
          throw new Error('Invalid JSON response from server')
        }
      }
      if (!responseData) {
        console.error('No response data received')
        throw new Error('No response data from server')
      }
      if (typeof responseData !== 'object') {
        console.error('Response data is not an object:', responseData)
        throw new Error(`Invalid response format: expected object, got ${typeof responseData}`)
      }
      if (!responseData.dataset_id && !responseData.filename) {
        console.error('Response missing essential fields:', responseData)
        throw new Error('Server response missing essential data (dataset_id or filename)')
      }
      const rowCount = responseData.rows || 0
      const successMessage = rowCount > 0 
        ? `✅ Complete! ${rowCount.toLocaleString()} rows processed`
        : '✅ Upload complete!'
      setActiveUploads(prev => {
        const newMap = new Map(prev)
        const uploadInfo = newMap.get(fileId)
        if (uploadInfo) {
          newMap.set(fileId, {
            ...uploadInfo,
            progress: 100,
            status: successMessage
          })
        }
        return newMap
      })
      const datasetData = {
        ...responseData,
        dataset_id: responseData.dataset_id || `dataset_${Date.now()}`
      }
      addDataset(datasetData)
      setTimeout(() => {
        setActiveUploads(prev => {
          const newMap = new Map(prev)
          newMap.delete(fileId)
          return newMap
        })
      }, 3000)
    } catch (error) {
      console.error('Upload error:', error)
      console.error('Error details:', {
        message: error.message,
        response: error.response,
        code: error.code,
        stack: error.stack
      })
      let errorMessage = 'Upload failed'
      if (error.message.includes('Invalid server response')) {
        errorMessage = `❌ Server Error: Invalid response format`
        console.error('Backend returned unexpected response format')
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = `❌ Timeout: File processing took too long`
      } else if (error.response?.status === 413) {
        errorMessage = `❌ File too large: Reduce file size`
      } else if (error.response?.status === 400) {
        const errorMsg = error.response.data?.error || 'Bad request'
        if (errorMsg.includes('encoding') || errorMsg.includes('decode')) {
          errorMessage = `❌ Encoding error: Save as UTF-8 CSV`
        } else {
          errorMessage = `❌ Format error: ${errorMsg.substring(0, 40)}...`
        }
      } else if (error.response?.data?.error) {
        const errorMsg = error.response.data.error
        errorMessage = `❌ Server: ${errorMsg.substring(0, 40)}...`
      } else if (error.request) {
        errorMessage = `❌ Network error: Check connection`
      } else {
        errorMessage = `❌ Error: ${error.message.substring(0, 40)}...`
      }
      setActiveUploads(prev => {
        const newMap = new Map(prev)
        const uploadInfo = newMap.get(fileId)
        if (uploadInfo) {
          newMap.set(fileId, {
            ...uploadInfo,
            progress: 0,
            status: errorMessage,
            error: true
          })
        }
        return newMap
      })
      setTimeout(() => {
        setActiveUploads(prev => {
          const newMap = new Map(prev)
          newMap.delete(fileId)
          return newMap
        })
      }, 5000)
    }
  }
  const onDrop = useCallback((acceptedFiles) => {
    acceptedFiles.forEach(uploadFile)
  }, [])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    multiple: true
  })
  const loadSampleData = async () => {
    const sampleDatasets = [
      {
        filename: 'credit_data_1.csv',
        data: [
          ['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN'],
          [100002, 1, 'Cash loans', 'M', 'N', 'Y', 0],
          [100003, 0, 'Cash loans', 'F', 'N', 'N', 0],
          [100004, 0, 'Revolving loans', 'M', 'Y', 'Y', 0],
          [100005, 1, 'Cash loans', 'F', 'N', 'Y', 1],
          [100006, 0, 'Cash loans', 'F', 'Y', 'N', 0]
        ]
      },
      {
        filename: 'credit_data_2.csv',
        data: [
          ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_INCOME_TYPE'],
          [100002, 202500.0, 406597.5, 24700.5, 351000.0, 'Working'],
          [100003, 270000.0, 1293502.5, 35698.5, 1129500.0, 'State servant'],
          [100004, 67500.0, 135000.0, 6750.0, 135000.0, 'Working'],
          [100005, 135000.0, 312682.5, 29686.5, 297000.0, 'Working'],
          [100006, 121500.0, 513000.0, 21865.5, 513000.0, 'Commercial associate']
        ]
      },
      {
        filename: 'credit_data_3.csv',
        data: [
          ['SK_ID_CURR', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION'],
          [100002, 2, 2, 10, 0],
          [100003, 1, 1, 11, 0],
          [100004, 2, 2, 9, 0],
          [100005, 1, 1, 17, 0],
          [100006, 2, 2, 14, 0]
        ]
      }
    ]
    const uploadPromises = sampleDatasets.map(dataset => {
      const csvContent = dataset.data.map(row => row.join(',')).join('\\n')
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const file = new File([blob], dataset.filename, { type: 'text/csv' })
      return uploadFile(file)
    })
    await Promise.allSettled(uploadPromises)
  }
  return (
    <div className="section">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center space-x-3 mb-6">
          <CloudArrowUpIcon className="w-8 h-8 text-blue-600" />
          <h2 className="text-3xl font-bold text-gray-900">Data Ingestion</h2>
        </div>
        <div className="mb-6 p-6 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl border border-blue-200 shadow-sm">
          <div className="flex items-center space-x-2 mb-3">
            <DocumentIcon className="w-5 h-5 text-blue-600" />
            <h4 className="font-semibold text-blue-900">Unlimited File Support</h4>
          </div>
          <div className="text-sm text-blue-800 space-y-2">
            <div className="flex items-center space-x-2">
              <CheckCircleIcon className="w-4 h-4 text-green-500" />
              <span><strong>File size:</strong> No limits! Upload files of any size</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircleIcon className="w-4 h-4 text-green-500" />
              <span><strong>Large Excel files:</strong> Automatically optimized processing</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircleIcon className="w-4 h-4 text-green-500" />
              <span><strong>Progress tracking:</strong> Real-time upload and server processing status</span>
            </div>
            <div className="flex items-center space-x-2">
              <ExclamationTriangleIcon className="w-4 h-4 text-amber-500" />
              <span><strong>Performance tip:</strong> CSV files process faster than Excel for large datasets</span>
            </div>
          </div>
        </div>
        {}
        <div 
          {...getRootProps()} 
          className={`border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
            isDragActive 
              ? 'border-blue-400 bg-blue-50' 
              : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          <CloudArrowUpIcon className="w-16 h-16 text-blue-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-700 mb-2">
            {isDragActive ? 'Drop files here...' : 'Upload Data Files'}
          </h3>
          <p className="text-gray-500 mb-4">
            Drag & drop CSV or Excel files here, or click to browse
          </p>
          <p className="text-sm text-gray-400">
            Supports: CSV, XLSX, XLS • No file size limits
          </p>
        </div>
        {}
        {activeUploads.size > 0 && (
          <div className="mt-6 space-y-3">
            <h4 className="font-medium text-gray-900">
              {activeUploads.size === 1 ? 'File Upload Progress' : `Uploading ${activeUploads.size} Files`}
            </h4>
            {Array.from(activeUploads.entries()).map(([fileId, uploadInfo]) => (
              <div key={fileId} className="p-4 bg-white rounded-lg border shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex-1 min-w-0">
                    <span className="text-sm font-medium text-gray-700 truncate block">
                      {uploadInfo.file.name}
                    </span>
                    <span className="text-xs text-gray-500">
                      {(uploadInfo.file.size / (1024 * 1024)).toFixed(1)}MB
                    </span>
                  </div>
                  <span className="text-sm text-gray-500 ml-4">
                    {uploadInfo.progress}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      uploadInfo.error ? 'bg-red-500' :
                      uploadInfo.progress === 100 ? 'bg-green-500' : 
                      uploadInfo.progress > 99 ? 'bg-orange-500' : 'bg-blue-600'
                    }`}
                    style={{ width: `${uploadInfo.progress}%` }}
                  ></div>
                </div>
                <div className="flex items-center text-sm">
                  {uploadInfo.progress > 99 && uploadInfo.progress < 100 && !uploadInfo.error && (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-orange-500 mr-2"></div>
                  )}
                  {uploadInfo.error && (
                    <XCircleIcon className="w-4 h-4 text-red-600 mr-2" />
                  )}
                  {uploadInfo.progress === 100 && !uploadInfo.error && (
                    <CheckCircleIcon className="w-4 h-4 text-green-600 mr-2" />
                  )}
                  <span className={`${
                    uploadInfo.error ? 'text-red-600' : 
                    uploadInfo.progress === 100 ? 'text-green-600' : 'text-gray-600'
                  }`}>
                    {uploadInfo.status}
                  </span>
                </div>
                {uploadInfo.file.size > 100 * 1024 * 1024 && uploadInfo.progress < 100 && (
                  <p className="text-xs text-blue-600 mt-2 font-medium">
                    🚀 Large file - optimized streaming upload in progress
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
        {}
        {datasets.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center space-x-2 mb-4">
              <DocumentIcon className="w-6 h-6 text-gray-700" />
              <h3 className="text-xl font-semibold text-gray-900">Uploaded Datasets</h3>
            </div>
            <div className="space-y-4">
              {datasets.map((dataset, index) => (
                <div key={index} className="bg-white rounded-lg shadow-sm border p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h4 className="font-semibold text-gray-900">{dataset.filename}</h4>
                      <p className="text-sm text-gray-500">
                        {dataset.rows.toLocaleString()} rows × {dataset.columns} columns
                      </p>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${
                        dataset.quality_report?.grade === 'A' ? 'text-green-600' :
                        dataset.quality_report?.grade === 'B' ? 'text-lime-600' :
                        dataset.quality_report?.grade === 'C' ? 'text-yellow-600' :
                        dataset.quality_report?.grade === 'D' ? 'text-orange-600' : 'text-red-600'
                      }`}>
                        {dataset.quality_report?.quality_score?.toFixed(1)} ({dataset.quality_report?.grade || 'N/A'})
                      </div>
                      <div className="text-sm text-gray-500">Quality Score</div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <div className="font-medium text-gray-700">Missing Values</div>
                      <div className="text-red-600">{dataset.quality_report?.issues?.missing_values || 0}</div>
                    </div>
                    <div>
                      <div className="font-medium text-gray-700">Duplicates</div>
                      <div className="text-yellow-600">{dataset.quality_report?.issues?.duplicates || 0}</div>
                    </div>
                    <div>
                      <div className="font-medium text-gray-700">Anomalies</div>
                      <div className="text-purple-600">{dataset.quality_report?.issues?.anomalies || 0}</div>
                    </div>
                    <div>
                      <div className="font-medium text-gray-700">Data Types</div>
                      <div className="text-blue-600">{dataset.quality_report?.basic_stats?.data_types || 'Mixed'}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
export default DataIngestion