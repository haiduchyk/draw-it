namespace Bellatrix.Utils.Editor
{
    using System;
    using System.Linq;
    using System.Reflection;
    using UnityEditor;
    using UnityEngine;

    [CustomEditor(typeof(MonoBehaviour), true)]
    public class EditorButton : Editor
    {
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            var mono = target as MonoBehaviour;

            var methods = mono.GetType()
                .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                .Where(o => Attribute.IsDefined(o, typeof(EditorButtonAttribute)));

            foreach (var method in methods)
            {
                if (GUILayout.Button(method.Name))
                {
                    method.Invoke(mono, null);
                }
            }
        }
    }
}
