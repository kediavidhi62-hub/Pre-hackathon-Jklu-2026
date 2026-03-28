import { ShieldCheck, ShieldAlert, ShieldQuestion, AlertTriangle, RotateCcw } from 'lucide-react'

const VERDICT_CONFIG = {
    FAKE: { icon: ShieldAlert, bg: 'bg-red-50', border: 'border-red-200', color: 'text-red-600', badge: 'bg-red-600', label: '⚠️ DEEPFAKE DETECTED' },
    REAL: { icon: ShieldCheck, bg: 'bg-green-50', border: 'border-green-200', color: 'text-green-600', badge: 'bg-green-600', label: '✅ AUTHENTIC' },
    SUSPICIOUS: { icon: ShieldQuestion, bg: 'bg-yellow-50', border: 'border-yellow-200', color: 'text-yellow-600', badge: 'bg-yellow-500', label: '🔍 SUSPICIOUS' },
}

function CircularGauge({ value, size = 160, strokeWidth = 12 }) {
    const radius = (size - strokeWidth) / 2
    const circumference = 2 * Math.PI * radius
    const offset = circumference - (value / 100) * circumference
    const color = value > 65 ? '#dc2626' : value > 35 ? '#eab308' : '#16a34a'

    return (
        <div className="relative inline-flex items-center justify-center">
            <svg width={size} height={size} className="-rotate-90">
                <circle cx={size / 2} cy={size / 2} r={radius} stroke="#e5e7eb" strokeWidth={strokeWidth} fill="none" />
                <circle
                    cx={size / 2} cy={size / 2} r={radius}
                    stroke={color} strokeWidth={strokeWidth}
                    fill="none" strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    className="transition-all duration-1000 ease-out"
                />
            </svg>
            <div className="absolute flex flex-col items-center justify-center text-center">
                <span className="text-4xl font-extrabold text-navy">{Math.round(value)}%</span>
                <p className="text-xs font-medium text-soft-gray uppercase tracking-wider mt-1">Fake Score</p>
            </div>
        </div>
    )
}

export default function ResultPanel({ result, onReset }) {
    const v = VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.SUSPICIOUS
    const percent = Math.round(result.score * 100)

    return (
        <section className="py-20 px-6">
            <div className="max-w-xl mx-auto space-y-6 animate-slide-up">

                {/* 1. Verdict & Probability Score */}
                <div className="grid sm:grid-cols-2 gap-6">
                    {/* Verdict Card */}
                    <div className={`bg-white border ${v.border} rounded-2xl p-8 shadow-sm flex flex-col items-center justify-center text-center relative overflow-hidden`}>
                        {/* Soft background glow matching verdict */}
                        <div className={`absolute -top-10 -right-10 w-32 h-32 rounded-full ${v.bg} blur-2xl opacity-50`} />
                        <div className={`absolute -bottom-10 -left-10 w-32 h-32 rounded-full ${v.bg} blur-2xl opacity-50`} />

                        <div className={`w-16 h-16 rounded-full ${v.bg} flex items-center justify-center mb-4 z-10`}>
                            {v.icon && <v.icon className={`w-8 h-8 ${v.color}`} />}
                        </div>
                        <p className="text-sm font-semibold text-soft-gray mb-1 z-10 uppercase tracking-wider">Final Verdict</p>
                        <h2 className={`text-2xl font-bold ${v.color} z-10 uppercase`}>{v.label.replace(/⚠️|✅|🔍/g, '').trim()}</h2>
                    </div>

                    {/* Probability Score Card */}
                    <div className="bg-white border border-gray-100 rounded-2xl p-8 shadow-sm flex flex-col items-center justify-center text-center">
                        <CircularGauge value={percent} size={140} />
                    </div>
                </div>

                {/* 2. Forensic Report (Explanation) */}
                <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
                    <h3 className="text-lg font-bold text-navy mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-brand-blue" />
                        Feedback
                    </h3>
                    <div className="p-4 bg-gray-50 rounded-xl border border-gray-100 min-h-[60px]">
                        <div className="text-soft-gray text-sm leading-relaxed whitespace-pre-line break-words overflow-wrap-anywhere">
                            {result.explanation || 'No explanation available.'}
                        </div>
                    </div>
                </div>

                {/* Reset Button */}
                <button
                    onClick={onReset}
                    className="w-full flex items-center justify-center gap-2 py-3.5 rounded-xl border-2 border-gray-200 text-navy font-semibold hover:border-brand-blue hover:text-brand-blue transition-all"
                >
                    <RotateCcw className="w-5 h-5" />
                    Scan Another File
                </button>
            </div>
        </section>
    )
}
