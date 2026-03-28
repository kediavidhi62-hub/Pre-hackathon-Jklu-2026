import { Film, FileDown, AlertTriangle, Code2, Gauge } from 'lucide-react'

const FEATURES = [
    { icon: Film, title: 'Frame-by-frame Video Analysis', description: 'Each frame is individually analyzed for temporal inconsistencies, flickering artifacts, and inter-frame anomalies.' },
    { icon: FileDown, title: 'Downloadable PDF Report', description: 'Generate comprehensive forensic PDF reports with visualizations, scores, and detailed AI explanations.' },
    { icon: AlertTriangle, title: 'Risk Impact Estimation', description: 'Understand the potential impact and spread risk of detected deepfakes with our AI risk scoring engine.' },
    { icon: Code2, title: 'API Access', description: 'Integrate DeepGuard detection into your own apps and workflows with our RESTful API.', badge: 'Coming Soon' },
    { icon: Gauge, title: 'Real-time Processing', description: 'Get results in seconds, not minutes. Our optimized AI pipeline delivers fast, accurate analysis.' },
]

export default function Features() {
    return (
        <section id="features" className="py-20 px-6">
            <div className="max-w-5xl mx-auto">
                {/* Section Header */}
                <div className="text-center mb-14">
                    <h2 className="text-3xl md:text-4xl font-bold text-navy mb-3">
                        Powerful <span className="gradient-text">Features</span>
                    </h2>
                    <p className="text-soft-gray max-w-xl mx-auto">
                        Everything you need for comprehensive deepfake analysis in one tool.
                    </p>
                </div>

                {/* Feature Grid — top row 3, bottom row 2 centered */}
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
                    {FEATURES.map((feature) => {
                        const Icon = feature.icon
                        return (
                            <div
                                key={feature.title}
                                className="group bg-white rounded-2xl border border-gray-100 shadow-sm p-6 hover:shadow-lg hover:-translate-y-1 transition-all duration-300 relative"
                            >
                                {feature.badge && (
                                    <span className="absolute top-4 right-4 text-[10px] font-bold px-2 py-0.5 rounded-full bg-purple-100 text-brand-purple">
                                        {feature.badge}
                                    </span>
                                )}
                                <div className="w-12 h-12 rounded-xl bg-blue-50 group-hover:bg-gradient-to-br group-hover:from-blue-100 group-hover:to-purple-100 flex items-center justify-center mb-4 transition-colors">
                                    <Icon className="w-6 h-6 text-brand-blue" />
                                </div>
                                <h3 className="font-semibold text-navy mb-2 text-sm">{feature.title}</h3>
                                <p className="text-soft-gray text-xs leading-relaxed">{feature.description}</p>
                            </div>
                        )
                    })}
                </div>
            </div>
        </section>
    )
}
