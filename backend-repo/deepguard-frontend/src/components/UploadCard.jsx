import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileWarning, Loader2 } from 'lucide-react'

const ACCEPTED = {
    'image/*': ['.jpg', '.jpeg', '.png', '.webp'],
    'video/*': ['.mp4', '.avi', '.mov', '.webm'],
}

const SCANNING_STEPS = [
    'Analyzing visual patterns…',
    'Running AI detection model…',
    'Evaluating authenticity…',
    'Generating feedback…',
]

export default function UploadCard({ onResult }) {
    const [selectedFile, setSelectedFile] = useState(null)
    const [scanning, setScanning] = useState(false)
    const [scanStep, setScanStep] = useState(0)

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            setSelectedFile(acceptedFiles[0])
        }
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: ACCEPTED,
        maxSize: 20 * 1024 * 1024,
        multiple: false,
    })

    const handleAnalyze = async () => {
        if (!selectedFile) return

        setScanning(true)
        setScanStep(0)
        const stepInterval = setInterval(() => {
            setScanStep(prev => Math.min(prev + 1, SCANNING_STEPS.length - 1))
        }, 1500)

        try {
            const formData = new FormData()
            formData.append('file', selectedFile)

            console.log('[DeepGuard] Sending file to backend:', selectedFile.name)

            // Create an AbortController for timeout (120s for Render cold-start + Gemini)
            const controller = new AbortController()
            const timeoutId = setTimeout(() => controller.abort(), 120000)

            const response = await fetch('https://pre-hackathon-jklu.onrender.com/analyze', {
                method: 'POST',
                body: formData,
                signal: controller.signal,
            })

            clearTimeout(timeoutId)
            console.log('[DeepGuard] Response status:', response.status)

            if (!response.ok) {
                const errorText = await response.text()
                console.error('[DeepGuard] Server error body:', errorText)
                throw new Error(`Server error: ${response.status}`)
            }

            const data = await response.json()
            console.log('[DeepGuard] Backend response:', data)

            clearInterval(stepInterval)
            setScanning(false)

            onResult({
                verdict: data.verdict?.toUpperCase() || 'SUSPICIOUS',
                score: data.score != null ? data.score : 0.5,
                explanation: data.explanation || data.report || 'Analysis complete. Manual review recommended.',
            })

        } catch (error) {
            console.error('[DeepGuard] API Error:', error)
            clearInterval(stepInterval)
            setScanning(false)

            if (error.name === 'AbortError') {
                alert('Request timed out. The backend server may be starting up (cold start). Please try again in 30 seconds.')
            } else {
                alert('Failed to connect to the backend. ' + error.message)
            }
        }
    }

    const handleReset = () => {
        setSelectedFile(null)
        setScanning(false)
        setScanStep(0)
    }

    /* ── Scanning State ── */
    if (scanning) {
        return (
            <section id="upload" className="py-20 px-6">
                <div className="max-w-lg mx-auto bg-white rounded-2xl shadow-xl shadow-black/5 border border-gray-100 p-10 text-center">
                    <Loader2 className="w-16 h-16 text-brand-blue animate-spin mx-auto mb-6" />
                    <p className="text-lg font-semibold text-navy mb-3">AI Analysis in Progress</p>
                    <div className="space-y-2">
                        {SCANNING_STEPS.map((step, i) => (
                            <div
                                key={step}
                                className={`flex items-center gap-2 text-sm transition-all duration-500 ${i <= scanStep ? 'text-navy opacity-100' : 'text-gray-300 opacity-50'
                                    }`}
                            >
                                <div className={`w-2 h-2 rounded-full transition-colors ${i < scanStep ? 'bg-green-500' : i === scanStep ? 'bg-brand-blue animate-pulse' : 'bg-gray-200'
                                    }`} />
                                {step}
                            </div>
                        ))}
                    </div>
                </div>
            </section>
        )
    }

    /* ── Upload State ── */
    return (
        <section id="upload" className="py-20 px-6">
            <div className="max-w-lg mx-auto">
                {/* Section Header */}
                <div className="text-center mb-10">
                    <h2 className="text-3xl md:text-4xl font-bold text-navy mb-3">
                        Upload & <span className="gradient-text">Analyze</span>
                    </h2>
                    <p className="text-soft-gray">
                        Drag a file or browse to start AI-powered deepfake detection.
                    </p>
                </div>

                {/* Card */}
                <div className="bg-white rounded-2xl shadow-xl shadow-black/5 border border-gray-100 overflow-hidden">
                    {/* Dropzone */}
                    <div className="p-6">
                        <div
                            {...getRootProps()}
                            className={`relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-300 ${isDragActive
                                ? 'border-brand-blue bg-blue-50 scale-[1.01]'
                                : 'border-gray-200 hover:border-brand-blue/50 hover:bg-blue-50/30'
                                }`}
                        >
                            <input {...getInputProps()} />
                            <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center transition-colors ${isDragActive ? 'bg-brand-blue/10' : 'bg-gray-50'
                                }`}>
                                <Upload className={`w-7 h-7 transition-colors ${isDragActive ? 'text-brand-blue' : 'text-soft-gray'}`} />
                            </div>
                            <p className="text-navy font-semibold mb-1">
                                {isDragActive ? 'Drop your file here…' : 'Drag & drop your file here'}
                            </p>
                            <p className="text-soft-gray text-sm mb-4">
                                or click to browse files
                            </p>
                            <p className="text-xs text-soft-gray/70">
                                Supports JPG, PNG, WEBP, MP4, AVI, MOV, WEBM (max 20MB)
                            </p>
                        </div>
                    </div>

                    {/* Selected File + Analyze */}
                    {selectedFile && (
                        <div className="px-6 pb-6 space-y-3 animate-fade-in">
                            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl">
                                <FileWarning className="w-5 h-5 text-brand-blue flex-shrink-0" />
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium text-navy truncate">{selectedFile.name}</p>
                                    <p className="text-xs text-soft-gray">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                                </div>
                                <button onClick={handleReset} className="text-xs text-soft-gray hover:text-red-500 transition-colors">
                                    Remove
                                </button>
                            </div>
                            <button
                                onClick={handleAnalyze}
                                className="w-full gradient-btn text-white font-semibold py-3 rounded-xl shadow-lg shadow-brand-blue/20 hover:shadow-xl hover:scale-[1.01] transition-all duration-300"
                            >
                                🔍 Analyze Now
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </section>
    )
}
