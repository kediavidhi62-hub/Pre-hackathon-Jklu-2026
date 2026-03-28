import { ScanFace, Grid3X3, Cpu, FileSearch } from 'lucide-react'

const ANALYSIS_ITEMS = [
    {
        icon: ScanFace,
        title: 'Facial Landmark Mismatch',
        description: 'Asymmetric feature alignment detected around the nose bridge and jawline. Distance ratios deviate from natural proportions.',
        severity: 'High',
        color: 'text-red-500',
    },
    {
        icon: Grid3X3,
        title: 'Pixel Inconsistency',
        description: 'Subtle blurring patterns found along facial boundaries. Compression artifacts are inconsistent with single-capture imagery.',
        severity: 'Medium',
        color: 'text-yellow-500',
    },
    {
        icon: Cpu,
        title: 'GAN Artifact Detection',
        description: 'Frequency-domain analysis reveals spectral patterns consistent with StyleGAN2 generative models.',
        severity: 'High',
        color: 'text-red-500',
    },
    {
        icon: FileSearch,
        title: 'Metadata Anomaly',
        description: 'EXIF data contains inconsistent device signatures. Editing timestamps suggest post-processing modifications.',
        severity: 'Low',
        color: 'text-green-500',
    },
]

export default function ExplainableAI() {
    return (
        <section className="py-20 px-6 bg-light-blue">
            <div className="max-w-6xl mx-auto">
                {/* Section Header */}
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold text-navy mb-3">
                        Explainable <span className="gradient-text">AI Analysis</span>
                    </h2>
                    <p className="text-soft-gray max-w-xl mx-auto">
                        Understand exactly why our AI made its decision — transparent and trustworthy.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 gap-10 items-start">
                    {/* Left: Heatmap Placeholder */}
                    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
                        <div className="relative aspect-square bg-gradient-to-br from-gray-100 to-gray-50 flex items-center justify-center">
                            {/* Simulated heatmap overlay */}
                            <div className="absolute inset-0 bg-gradient-to-br from-red-500/20 via-yellow-500/10 to-green-500/5 rounded-lg" />
                            <div className="absolute top-[30%] left-[35%] w-24 h-24 rounded-full bg-red-500/30 blur-xl animate-pulse" />
                            <div className="absolute top-[40%] left-[45%] w-16 h-16 rounded-full bg-red-500/40 blur-lg animate-pulse" />
                            <div className="absolute bottom-[30%] right-[30%] w-20 h-20 rounded-full bg-yellow-500/20 blur-xl" />
                            <ScanFace className="w-32 h-32 text-gray-300 relative z-10" />
                        </div>
                        <div className="p-4 text-center">
                            <p className="text-sm text-soft-gray">Heatmap overlay showing areas of detected manipulation</p>
                        </div>
                    </div>

                    {/* Right: Breakdown */}
                    <div className="space-y-4">
                        {ANALYSIS_ITEMS.map((item, i) => {
                            const Icon = item.icon
                            return (
                                <div key={i} className="bg-white rounded-xl border border-gray-100 shadow-sm p-5 hover:shadow-md transition-shadow">
                                    <div className="flex items-start gap-3">
                                        <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center flex-shrink-0">
                                            <Icon className="w-5 h-5 text-brand-blue" />
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="font-semibold text-navy text-sm">{item.title}</h4>
                                                <span className={`text-xs font-medium ${item.color}`}>
                                                    {item.severity}
                                                </span>
                                            </div>
                                            <p className="text-soft-gray text-sm leading-relaxed">{item.description}</p>
                                        </div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            </div>
        </section>
    )
}
