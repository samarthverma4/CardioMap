import { OrbitControls, useGLTF } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Evidence, Prediction, Vessel } from '../types'
import { VESSELS } from '../types'
import { pct } from '../api'

/* Heart: textured surface "Realistic Human Heart" by neshallads (CC BY 4.0) with the BodyParts3D coronary
   arteries (DBCLS, CC BY 4.0) fitted onto it; built by tools/build_heart.py + tools/fit_surface.py.
   The surface is split into each artery's territory (myo-LAD, myo-LCX, myo-RCA) plus myo-base. */
const MODEL_URL = '/heart.glb'
const OPACITY: Record<Evidence, number> = { moderate: 1, limited: 0.7, weak: 0.45 }
const VESSEL = '#86939c'

type Nodes = Record<string, THREE.Mesh>

/* Weak evidence: a dashed outline drawn as a slightly inflated back-face shell with stripes discarded. */
const dashedOutline = (color: string) =>
  new THREE.ShaderMaterial({
    uniforms: { color: { value: new THREE.Color(color) } },
    vertexShader: `
      varying vec3 vPos;
      void main() {
        vec3 p = position + normal * 0.022;
        vPos = p;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: `
      uniform vec3 color;
      varying vec3 vPos;
      void main() {
        if (fract((vPos.x + vPos.y + vPos.z) * 9.0) > 0.5) discard;
        gl_FragColor = vec4(mix(color, vec3(0.92), 0.55), 0.9);
      }`,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
  })

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
      <spriteMaterial map={tex} transparent />
    </sprite>
  )
}

/* Label anchor: the artery's most characteristic point, pushed outward from the heart's centre.
   RCA: most rightward (right AV groove); LCX: most lateral (left AV groove); LAD: most anterior. */
const ANCHOR_DIR: Record<Vessel, THREE.Vector3> = {
  LAD: new THREE.Vector3(0.3, 0, 1),
  LCX: new THREE.Vector3(1, 0.25, 0),
  RCA: new THREE.Vector3(-1, 0.2, 0.3),
}
function labelAnchor(geo: THREE.BufferGeometry, vessel: Vessel) {
  const pos = geo.attributes.position as THREE.BufferAttribute
  const dir = ANCHOR_DIR[vessel].clone().normalize()
  const v = new THREE.Vector3(), best = new THREE.Vector3()
  let bestDot = -Infinity
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i)
    const d = v.dot(dir)
    if (d > bestDot) { bestDot = d; best.copy(v) }
  }
  return best.add(dir.multiplyScalar(0.2))
}

/* Contrast-fill animation: each artery vertex gets aDist = distance from the vessel's ostium (0) to its
   farthest branch (1). A uniform uFill sweeps 0 -> 1: behind it the vessel shows its colour, ahead of it the
   vessel is dark and empty, and a bright front marks the dye. Material colour itself never changes. */
const FILL_MS = 1200
const FILL_END = 1.08
const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

function withFillDistance(geo: THREE.BufferGeometry, ostium: THREE.Vector3) {
  if (geo.getAttribute('aDist')) return geo
  const pos = geo.attributes.position as THREE.BufferAttribute
  const d = new Float32Array(pos.count)
  const v = new THREE.Vector3()
  let max = 1e-6
  for (let i = 0; i < pos.count; i++) {
    d[i] = v.fromBufferAttribute(pos, i).distanceTo(ostium)
    max = Math.max(max, d[i])
  }
  for (let i = 0; i < d.length; i++) d[i] /= max
  geo.setAttribute('aDist', new THREE.BufferAttribute(d, 1))
  return geo
}

/* Ostium: LAD and LCX start where they meet the left main; the RCA starts at its highest point (aortic root). */
function ostiumOf(vessel: Vessel, geo: THREE.BufferGeometry, lm: THREE.BufferGeometry) {
  const pos = geo.attributes.position as THREE.BufferAttribute
  const v = new THREE.Vector3(), best = new THREE.Vector3()
  if (vessel === 'RCA') {
    let top = -Infinity
    for (let i = 0; i < pos.count; i++) { v.fromBufferAttribute(pos, i); if (v.y > top) { top = v.y; best.copy(v) } }
    return best
  }
  lm.computeBoundingBox()
  const c = lm.boundingBox!.getCenter(new THREE.Vector3())
  let near = Infinity
  for (let i = 0; i < pos.count; i++) { v.fromBufferAttribute(pos, i); const dd = v.distanceTo(c); if (dd < near) { near = dd; best.copy(v) } }
  return best
}

function fillShader(uniforms: { uFill: { value: number } }) {
  return (shader: { uniforms: Record<string, unknown>; vertexShader: string; fragmentShader: string }) => {
    shader.uniforms.uFill = uniforms.uFill
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float aDist;\nvarying float vDist;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvDist = aDist;')
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uFill;\nvarying float vDist;')
      .replace('#include <color_fragment>', `#include <color_fragment>
        float filled = step(vDist, uFill);
        diffuseColor.rgb = mix(vec3(0.16, 0.12, 0.12), diffuseColor.rgb, filled);`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        totalEmissiveRadiance *= filled;
        float front = smoothstep(0.07, 0.0, abs(vDist - uFill)) * step(uFill, 1.0);
        totalEmissiveRadiance += vec3(1.0, 0.94, 0.82) * front * 1.6;`)
  }
}

interface ArteryProps {
  vessel: Vessel
  geometry: THREE.BufferGeometry
  color: string
  evidence: Evidence
  selected: boolean
  fillTrigger: number
  onTouch: (v: Vessel) => void
  onHover: (v: Vessel | null, e?: ThreeEvent<PointerEvent>) => void
  onSelect: (v: Vessel) => void
}

function Artery({ vessel, geometry, color, evidence, selected, fillTrigger, onTouch, onHover, onSelect }: ArteryProps) {
  const opacity = OPACITY[evidence]
  const uniforms = useMemo(() => ({ uFill: { value: FILL_END } }), [])
  const onBeforeCompile = useMemo(() => fillShader(uniforms), [uniforms])
  const start = useRef<number | null>(null)
  const invalidate = useThree((st) => st.invalidate)
  useEffect(() => {
    if (fillTrigger === 0 || reducedMotion()) return
    start.current = performance.now()
    uniforms.uFill.value = 0
    invalidate()
  }, [fillTrigger, uniforms, invalidate])
  useFrame(() => {
    if (start.current === null) return
    const t = Math.min((performance.now() - start.current) / FILL_MS, 1)
    uniforms.uFill.value = (1 - Math.pow(1 - t, 2)) * FILL_END   // ease-out, like dye slowing distally
    if (t >= 1) start.current = null
    invalidate()
  })
  const outline = useMemo(() => (evidence === 'weak' ? dashedOutline(color) : null), [evidence, color])
  useEffect(() => () => outline?.dispose(), [outline])
  const anchor = useMemo(() => labelAnchor(geometry, vessel), [geometry, vessel])
  return (
    <group>
      <mesh
        name={vessel}
        geometry={geometry}
        scale={selected ? 1.012 : 1}
        onPointerOver={(e) => { e.stopPropagation(); onHover(vessel, e) }}
        onPointerMove={(e) => { e.stopPropagation(); onHover(vessel, e) }}
        onPointerOut={() => onHover(null)}
        onPointerDown={(e) => { e.stopPropagation(); onTouch(vessel) }}
        onClick={(e) => { e.stopPropagation(); onSelect(vessel) }}
      >
        <meshStandardMaterial
          onBeforeCompile={onBeforeCompile}
          customProgramCacheKey={() => 'artery-fill'}
          color={color}
          emissive={color}
          emissiveIntensity={selected ? 0.55 : 0.35}
          roughness={0.35}
          transparent={opacity < 1}
          opacity={opacity}
          depthWrite={opacity === 1}
        />
      </mesh>
      {outline && <mesh geometry={geometry} material={outline} name={`${vessel}-weak-outline`} raycast={() => null} />}
      <Label text={vessel} position={anchor} selected={selected} />
    </group>
  )
}

function Myocardium({ nodes, tint }: { nodes: Nodes; tint: { vessel: Vessel; color: string } | null }) {
  // Surface meshes carry their own (textured) material from the GLB; the selected territory is tinted by
  // multiplying the texture with the vessel colour.
  const tinted = useMemo(() => {
    if (!tint) return null
    const m = (nodes[`myo-${tint.vessel}`].material as THREE.MeshStandardMaterial).clone()
    m.color = new THREE.Color('#ffffff').lerp(new THREE.Color(tint.color), 0.45)
    m.emissive = new THREE.Color(tint.color)
    m.emissiveIntensity = 0.06
    return m
  }, [nodes, tint])
  useEffect(() => () => tinted?.dispose(), [tinted])
  const surface = ['myo-LAD', 'myo-LCX', 'myo-RCA', 'myo-base']
  return (
    <group>
      {surface.map((n) => (
        <mesh key={n} name={n} geometry={nodes[n].geometry}
          material={tint && n === `myo-${tint.vessel}` ? tinted! : nodes[n].material} />
      ))}
      {['aorta', 'pulmonary'].filter((n) => nodes[n]).map((n) => (
        <mesh key={n} name={n} geometry={nodes[n].geometry}>
          <meshStandardMaterial color={VESSEL} roughness={0.6} />
        </mesh>
      ))}
      <mesh name="LM" geometry={nodes.LM.geometry}>
        <meshStandardMaterial color="#9aa4ab" roughness={0.5} />
      </mesh>
    </group>
  )
}

/* Narrow (portrait) canvases: pull the camera back and lift the heart clear of the legend. */
function CameraFit() {
  const { camera, size, invalidate } = useThree()
  useEffect(() => {
    const portrait = size.width / size.height < 0.9
    camera.position.setLength(portrait ? 9.6 : 7.0)
    camera.updateProjectionMatrix()
    invalidate()
  }, [camera, size.width, size.height, invalidate])
  return null
}

function SceneProbe() {
  const scene = useThree((s) => s.scene)
  const size = useThree((s) => s.size)
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => invalidate(), [size, invalidate]) // demand frameloop: redraw after canvas resize
  useEffect(() => {
    // Read by tests/test_scene_color.py: the colour actually set on each artery mesh.
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

interface SceneProps {
  prediction: Prediction | null
  selected: Vessel | null
  onSelect: (v: Vessel | null) => void
  tint: { vessel: Vessel; color: string } | null
  onHover: (v: Vessel | null, e?: ThreeEvent<PointerEvent>) => void
}

function HeartModel({ prediction, selected, onSelect, tint, onHover }: SceneProps) {
  const { nodes } = useGLTF(MODEL_URL, false, false) as unknown as { nodes: Nodes }
  const state = prediction?.scene_state
  const geos = useMemo(
    () => Object.fromEntries(VESSELS.map((v) => [v, withFillDistance(nodes[v].geometry, ostiumOf(v, nodes[v].geometry, nodes.LM.geometry))])) as Record<Vessel, THREE.BufferGeometry>,
    [nodes],
  )
  // Re-run the fill whenever a vessel is touched in the scene or selected anywhere (list, bullseye).
  const [fills, setFills] = useState<Record<Vessel, number>>({ LAD: 0, LCX: 0, RCA: 0 })
  const fill = (v: Vessel) => setFills((f) => ({ ...f, [v]: f[v] + 1 }))
  useEffect(() => { if (selected) fill(selected) }, [selected])
  return (
    <group rotation={[0.05, -0.2, 0]} position={[0, -0.1, 0]}>
      <Myocardium nodes={nodes} tint={tint} />
      {state &&
        VESSELS.map((v) => (
          <Artery
            key={v}
            vessel={v}
            geometry={geos[v]}
            color={state[v].color}
            evidence={state[v].evidence}
            selected={selected === v}
            fillTrigger={fills[v]}
            onTouch={fill}
            onHover={onHover}
            onSelect={onSelect}
          />
        ))}
    </group>
  )
}

interface HeartProps {
  prediction: Prediction | null
  selected: Vessel | null
  onSelect: (v: Vessel | null) => void
  showTerritory: boolean
}

export default function Heart({ prediction, selected, onSelect, showTerritory }: HeartProps) {
  const [hover, setHover] = useState<{ v: Vessel; x: number; y: number } | null>(null)
  const state = prediction?.scene_state
  const tint = showTerritory && selected && state ? { vessel: selected, color: state[selected].color } : null
  useEffect(() => {
    document.body.style.cursor = hover ? 'pointer' : ''
  }, [hover])

  return (
    <>
      <Canvas
        frameloop="demand"
        style={{ position: 'absolute', inset: 0 }}
        dpr={[1, 1.5]}
        camera={{ position: [0, 0.1, 8.2], fov: 36 }}
        onPointerMissed={() => onSelect(null)}
        aria-label="Anatomical 3D heart with coronary arteries coloured by predicted probability"
      >
        <hemisphereLight args={['#f2e6e2', '#2a3640', 0.9]} />
        <directionalLight position={[-3, 4, 5]} intensity={1.7} />
        <directionalLight position={[3, -2, -4]} intensity={0.8} color="#9fc2e0" />
        <HeartModel
          prediction={prediction}
          selected={selected}
          onSelect={onSelect}
          tint={tint}
          onHover={(v, e) => setHover(v && e ? { v, x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY } : null)}
        />
        <OrbitControls enablePan={false} minDistance={4} maxDistance={12} makeDefault />
        <CameraFit />
        <SceneProbe />
      </Canvas>
      {hover && prediction && (
        <div className="scene-tip" style={{ left: hover.x, top: hover.y }} role="status">
          <strong>{hover.v}</strong> {pct(prediction.vessels[hover.v].p)}
          <span> · {prediction.vessels[hover.v].evidence} evidence</span>
          <span> · P(CAD) {pct(prediction.cad.p)}</span>
        </div>
      )}
    </>
  )
}

useGLTF.preload(MODEL_URL, false, false)
