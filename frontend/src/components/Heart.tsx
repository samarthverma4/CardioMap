import { Html, Line, OrbitControls } from '@react-three/drei'
import { Canvas, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'
import type { Evidence, Prediction, Vessel } from '../types'
import { VESSELS } from '../types'
import { pct } from '../api'

/* Generic, procedural heart. Ventricles = one tapered ellipsoid; every artery point and territory
   vertex is computed from the same surface function, so vessels always sit on the myocardium.
   Angles: lon 0 = anterior, +90 = patient's left (lateral LV), +/-180 = posterior. */
const A = 1.0, B = 1.3, C = 0.9
const D2R = Math.PI / 180

function shape(x: number, y: number, z: number) {
  const lon = Math.atan2(x, z) / D2R
  const taper = y < 0 ? 1 + 0.38 * y : 1
  const rvBulge = 1 + 0.07 * Math.exp(-(((lon + 60) / 40) ** 2)) * (y > -0.6 ? 1 : 0)
  return new THREE.Vector3(A * x * taper * rvBulge, B * y, C * z * taper * rvBulge)
}

function surf(lat: number, lon: number, lift = 0.035) {
  const la = lat * D2R, lo = lon * D2R
  return shape(Math.cos(la) * Math.sin(lo), Math.sin(la), Math.cos(la) * Math.cos(lo)).multiplyScalar(1 + lift)
}

const PATHS: Record<Vessel, [number, number][]> = {
  // anterior interventricular groove to the apex, wrapping slightly
  LAD: [[48, 2], [40, -10], [28, -12], [12, -8], [-5, -4], [-25, 0], [-45, 4], [-63, 8], [-77, 18], [-86, 50]],
  // left AV groove round to the back
  LCX: [[48, 2], [42, 25], [38, 45], [36, 65], [35, 85], [34, 105], [32, 125], [29, 145], [24, 160]],
  // right AV groove round to the posterior groove, then down to the inferior wall
  RCA: [[42, -38], [37, -58], [35, -80], [34, -105], [34, -130], [33, -155], [30, -178], [18, -184], [2, -186], [-16, -186], [-32, -184]],
}
const ORIGINS: Record<string, THREE.Vector3[]> = {
  LM: [new THREE.Vector3(-0.05, 1.18, 0.42), surf(48, 2)],
  RCA: [new THREE.Vector3(-0.28, 1.12, 0.42), surf(42, -38)],
}

function territory(lat: number, lon: number): Vessel | null {
  if (lat > 34) return null
  if (lat < -66) return 'LAD'
  if (lon >= -40 && lon < 50) return 'LAD'
  if (lon >= 50 && lon < 140) return 'LCX'
  return 'RCA'
}

function makeLabel(text: string, selected: boolean) {
  const c = document.createElement('canvas')
  c.width = 128; c.height = 56
  const g = c.getContext('2d')!
  g.fillStyle = selected ? 'rgba(232,238,242,0.95)' : 'rgba(8,13,17,0.72)'
  g.beginPath(); g.roundRect(8, 8, 112, 40, 9); g.fill()
  g.font = '600 26px "IBM Plex Mono", ui-monospace, monospace'
  g.fillStyle = selected ? '#111a21' : '#e8eef2'
  g.textAlign = 'center'; g.textBaseline = 'middle'
  g.fillText(text, 64, 29)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

function Label({ text, position, selected }: { text: string; position: THREE.Vector3; selected: boolean }) {
  const tex = useMemo(() => makeLabel(text, selected), [text, selected])
  useEffect(() => () => tex.dispose(), [tex])
  return (
    <sprite position={position} scale={[0.46, 0.2, 1]} renderOrder={10} raycast={() => null}>
      <spriteMaterial map={tex} depthTest={false} transparent />
    </sprite>
  )
}

const OPACITY: Record<Evidence, number> = { moderate: 1, limited: 0.7, weak: 0.45 }
const MYO = new THREE.Color('#71818c')

function Ventricles({ tint }: { tint: { vessel: Vessel; color: string } | null }) {
  const geo = useMemo(() => {
    const g = new THREE.SphereGeometry(1, 56, 36)
    const pos = g.attributes.position
    const v = new THREE.Vector3()
    const terr: (Vessel | null)[] = []
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i)
      const lat = Math.asin(THREE.MathUtils.clamp(v.y, -1, 1)) / D2R
      const lon = Math.atan2(v.x, v.z) / D2R
      terr.push(territory(lat, lon))
      const s = shape(v.x, v.y, v.z)
      pos.setXYZ(i, s.x, s.y, s.z)
    }
    g.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(pos.count * 3), 3))
    g.computeVertexNormals()
    g.userData.territory = terr
    return g
  }, [])

  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    const col = geo.attributes.color as THREE.BufferAttribute
    const terr = geo.userData.territory as (Vessel | null)[]
    const t = tint ? new THREE.Color(tint.color) : null
    const c = new THREE.Color()
    for (let i = 0; i < col.count; i++) {
      c.copy(MYO)
      if (t && terr[i] === tint!.vessel) c.lerp(t, 0.45)
      col.setXYZ(i, c.r, c.g, c.b)
    }
    col.needsUpdate = true
    invalidate()
  }, [geo, tint, invalidate])

  return (
    <mesh geometry={geo} name="myocardium">
      <meshStandardMaterial vertexColors roughness={0.75} metalness={0.05} />
    </mesh>
  )
}

function Tube({ points, radius, color, name }: { points: THREE.Vector3[]; radius: number; color: string; name?: string }) {
  const geo = useMemo(() => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 24, radius, 10, false), [points, radius])
  return (
    <mesh geometry={geo} name={name}>
      <meshStandardMaterial color={color} roughness={0.6} />
    </mesh>
  )
}

function GreatVessels() {
  const aorta = useMemo(() => [
    [-0.12, 0.95, 0.3], [-0.15, 1.55, 0.28], [0.05, 2.0, 0.02], [0.5, 2.0, -0.35], [0.66, 1.45, -0.6],
  ].map((p) => new THREE.Vector3(...p)), [])
  const pulm = useMemo(() => [[0.32, 0.92, 0.52], [0.38, 1.55, 0.5], [0.3, 1.95, 0.12]].map((p) => new THREE.Vector3(...p)), [])
  return (
    <group>
      <mesh position={[0.45, 1.12, -0.45]} scale={[1, 0.8, 1]}>
        <sphereGeometry args={[0.52, 24, 16]} />
        <meshStandardMaterial color="#64737e" roughness={0.8} />
      </mesh>
      <mesh position={[-0.82, 1.22, 0.02]} scale={[1, 0.85, 1]}>
        <sphereGeometry args={[0.46, 24, 16]} />
        <meshStandardMaterial color="#64737e" roughness={0.8} />
      </mesh>
      <Tube points={aorta} radius={0.21} color="#7d8b95" />
      <Tube points={pulm} radius={0.19} color="#74838d" />
    </group>
  )
}

interface ArteryProps {
  vessel: Vessel
  color: string
  evidence: Evidence
  selected: boolean
  onHover: (v: Vessel | null, at?: THREE.Vector3) => void
  onSelect: (v: Vessel) => void
}

function Artery({ vessel, color, evidence, selected, onHover, onSelect }: ArteryProps) {
  const curve = useMemo(() => {
    const pts = PATHS[vessel].map(([la, lo]) => surf(la, lo))
    if (vessel === 'RCA') pts.unshift(ORIGINS.RCA[0])
    return new THREE.CatmullRomCurve3(pts)
  }, [vessel])
  const r = selected ? 0.062 : 0.045
  const geo = useMemo(() => new THREE.TubeGeometry(curve, 72, r, 8, false), [curve, r])
  const dash = useMemo(() => curve.getPoints(90).map((p) => p.clone().multiplyScalar(1.075)), [curve])
  const opacity = OPACITY[evidence]
  const mid = useMemo(() => curve.getPoint(vessel === 'LAD' ? 0.45 : 0.55).multiplyScalar(1.12), [curve, vessel])
  return (
    <group>
      <mesh
        name={vessel}
        geometry={geo}
        onPointerOver={(e) => { e.stopPropagation(); onHover(vessel, e.point.clone()) }}
        onPointerOut={() => onHover(null)}
        onClick={(e) => { e.stopPropagation(); onSelect(vessel) }}
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.35}
          roughness={0.35}
          transparent={opacity < 1}
          opacity={opacity}
          depthWrite={opacity === 1}
        />
      </mesh>
      {evidence === 'weak' && (
        <Line points={dash} color="#e8eef2" lineWidth={1.6} dashed dashSize={0.08} gapSize={0.07} transparent opacity={0.85} name={`${vessel}-weak-outline`} />
      )}
      <Label text={vessel} position={mid} selected={selected} />
    </group>
  )
}

function SceneProbe() {
  const scene = useThree((s) => s.scene)
  const size = useThree((s) => s.size)
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => invalidate(), [size, invalidate]) // demand frameloop: redraw after canvas resize
  useEffect(() => {
    // Read by tests/test_scene_color: the colour actually rendered on each artery mesh.
    ;(window as unknown as { __cardiomap: unknown }).__cardiomap = {
      arteryColors: () =>
        Object.fromEntries(VESSELS.map((v) => {
          const m = scene.getObjectByName(v) as THREE.Mesh | undefined
          return [v, m ? '#' + (m.material as THREE.MeshStandardMaterial).color.getHexString() : null]
        })),
    }
  }, [scene])
  return null
}

interface HeartProps {
  prediction: Prediction | null
  selected: Vessel | null
  onSelect: (v: Vessel | null) => void
  showTerritory: boolean
}

export default function Heart({ prediction, selected, onSelect, showTerritory }: HeartProps) {
  const [hover, setHover] = useState<{ v: Vessel; at: THREE.Vector3 } | null>(null)
  const state = prediction?.scene_state
  const tint = showTerritory && selected && state ? { vessel: selected, color: state[selected].color } : null
  useEffect(() => {
    document.body.style.cursor = hover ? 'pointer' : ''
  }, [hover])

  return (
    <Canvas
      frameloop="demand"
      dpr={[1, 1.5]}
      camera={{ position: [0.3, 0.6, 7.6], fov: 36 }}
      onPointerMissed={() => onSelect(null)}
      aria-label="Generic 3D heart with coronary arteries coloured by predicted probability"
    >
      <ambientLight intensity={0.55} />
      <directionalLight position={[-3, 4, 5]} intensity={1.6} />
      <directionalLight position={[3, -2, -4]} intensity={0.7} color="#9fc2e0" />
      <group rotation={[0.12, 0.28, 0.5]} position={[0, -0.25, 0]}>
        <Ventricles tint={tint} />
        <GreatVessels />
        <Tube points={ORIGINS.LM} radius={0.05} color="#b9c4cb" name="LM" />
        {state &&
          VESSELS.map((v) => (
            <Artery
              key={v}
              vessel={v}
              color={state[v].color}
              evidence={state[v].evidence}
              selected={selected === v}
              onHover={(vv, at) => setHover(vv && at ? { v: vv, at } : null)}
              onSelect={onSelect}
            />
          ))}
      </group>
      {hover && prediction && (
        <Html position={hover.at} className="scene-tip" style={{ pointerEvents: 'none' }}>
          <div>
            <strong>{hover.v}</strong> {pct(prediction.vessels[hover.v].p)}
            <span> · {prediction.vessels[hover.v].evidence} evidence</span>
            <span> · P(CAD) {pct(prediction.cad.p)}</span>
          </div>
        </Html>
      )}
      <OrbitControls enablePan={false} minDistance={4} maxDistance={12} target={[0, 0.25, 0]} makeDefault />
      <SceneProbe />
    </Canvas>
  )
}
