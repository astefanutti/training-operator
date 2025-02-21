// Copyright 2024 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated by applyconfiguration-gen. DO NOT EDIT.

package v1alpha1

import (
	v1 "k8s.io/api/core/v1"
)

// TrainerApplyConfiguration represents a declarative configuration of the Trainer type for use
// with apply.
type TrainerApplyConfiguration struct {
	Image            *string                  `json:"image,omitempty"`
	Command          []string                 `json:"command,omitempty"`
	Args             []string                 `json:"args,omitempty"`
	Env              []v1.EnvVar              `json:"env,omitempty"`
	NumNodes         *int32                   `json:"numNodes,omitempty"`
	ResourcesPerNode *v1.ResourceRequirements `json:"resourcesPerNode,omitempty"`
	NumProcPerNode   *string                  `json:"numProcPerNode,omitempty"`
}

// TrainerApplyConfiguration constructs a declarative configuration of the Trainer type for use with
// apply.
func Trainer() *TrainerApplyConfiguration {
	return &TrainerApplyConfiguration{}
}

// WithImage sets the Image field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the Image field is set to the value of the last call.
func (b *TrainerApplyConfiguration) WithImage(value string) *TrainerApplyConfiguration {
	b.Image = &value
	return b
}

// WithCommand adds the given value to the Command field in the declarative configuration
// and returns the receiver, so that objects can be build by chaining "With" function invocations.
// If called multiple times, values provided by each call will be appended to the Command field.
func (b *TrainerApplyConfiguration) WithCommand(values ...string) *TrainerApplyConfiguration {
	for i := range values {
		b.Command = append(b.Command, values[i])
	}
	return b
}

// WithArgs adds the given value to the Args field in the declarative configuration
// and returns the receiver, so that objects can be build by chaining "With" function invocations.
// If called multiple times, values provided by each call will be appended to the Args field.
func (b *TrainerApplyConfiguration) WithArgs(values ...string) *TrainerApplyConfiguration {
	for i := range values {
		b.Args = append(b.Args, values[i])
	}
	return b
}

// WithEnv adds the given value to the Env field in the declarative configuration
// and returns the receiver, so that objects can be build by chaining "With" function invocations.
// If called multiple times, values provided by each call will be appended to the Env field.
func (b *TrainerApplyConfiguration) WithEnv(values ...v1.EnvVar) *TrainerApplyConfiguration {
	for i := range values {
		b.Env = append(b.Env, values[i])
	}
	return b
}

// WithNumNodes sets the NumNodes field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the NumNodes field is set to the value of the last call.
func (b *TrainerApplyConfiguration) WithNumNodes(value int32) *TrainerApplyConfiguration {
	b.NumNodes = &value
	return b
}

// WithResourcesPerNode sets the ResourcesPerNode field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the ResourcesPerNode field is set to the value of the last call.
func (b *TrainerApplyConfiguration) WithResourcesPerNode(value v1.ResourceRequirements) *TrainerApplyConfiguration {
	b.ResourcesPerNode = &value
	return b
}

// WithNumProcPerNode sets the NumProcPerNode field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the NumProcPerNode field is set to the value of the last call.
func (b *TrainerApplyConfiguration) WithNumProcPerNode(value string) *TrainerApplyConfiguration {
	b.NumProcPerNode = &value
	return b
}
