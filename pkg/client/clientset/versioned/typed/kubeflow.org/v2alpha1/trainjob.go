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

// Code generated by client-gen. DO NOT EDIT.

package v2alpha1

import (
	"context"
	json "encoding/json"
	"fmt"
	"time"

	v2alpha1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v2alpha1"
	kubefloworgv2alpha1 "github.com/kubeflow/training-operator/pkg/client/applyconfiguration/kubeflow.org/v2alpha1"
	scheme "github.com/kubeflow/training-operator/pkg/client/clientset/versioned/scheme"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
)

// TrainJobsGetter has a method to return a TrainJobInterface.
// A group's client should implement this interface.
type TrainJobsGetter interface {
	TrainJobs(namespace string) TrainJobInterface
}

// TrainJobInterface has methods to work with TrainJob resources.
type TrainJobInterface interface {
	Create(ctx context.Context, trainJob *v2alpha1.TrainJob, opts v1.CreateOptions) (*v2alpha1.TrainJob, error)
	Update(ctx context.Context, trainJob *v2alpha1.TrainJob, opts v1.UpdateOptions) (*v2alpha1.TrainJob, error)
	UpdateStatus(ctx context.Context, trainJob *v2alpha1.TrainJob, opts v1.UpdateOptions) (*v2alpha1.TrainJob, error)
	Delete(ctx context.Context, name string, opts v1.DeleteOptions) error
	DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error
	Get(ctx context.Context, name string, opts v1.GetOptions) (*v2alpha1.TrainJob, error)
	List(ctx context.Context, opts v1.ListOptions) (*v2alpha1.TrainJobList, error)
	Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error)
	Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v2alpha1.TrainJob, err error)
	Apply(ctx context.Context, trainJob *kubefloworgv2alpha1.TrainJobApplyConfiguration, opts v1.ApplyOptions) (result *v2alpha1.TrainJob, err error)
	ApplyStatus(ctx context.Context, trainJob *kubefloworgv2alpha1.TrainJobApplyConfiguration, opts v1.ApplyOptions) (result *v2alpha1.TrainJob, err error)
	TrainJobExpansion
}

// trainJobs implements TrainJobInterface
type trainJobs struct {
	client rest.Interface
	ns     string
}

// newTrainJobs returns a TrainJobs
func newTrainJobs(c *KubeflowV2alpha1Client, namespace string) *trainJobs {
	return &trainJobs{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Get takes name of the trainJob, and returns the corresponding trainJob object, and an error if there is any.
func (c *trainJobs) Get(ctx context.Context, name string, options v1.GetOptions) (result *v2alpha1.TrainJob, err error) {
	result = &v2alpha1.TrainJob{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("trainjobs").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do(ctx).
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of TrainJobs that match those selectors.
func (c *trainJobs) List(ctx context.Context, opts v1.ListOptions) (result *v2alpha1.TrainJobList, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &v2alpha1.TrainJobList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("trainjobs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Do(ctx).
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested trainJobs.
func (c *trainJobs) Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("trainjobs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Watch(ctx)
}

// Create takes the representation of a trainJob and creates it.  Returns the server's representation of the trainJob, and an error, if there is any.
func (c *trainJobs) Create(ctx context.Context, trainJob *v2alpha1.TrainJob, opts v1.CreateOptions) (result *v2alpha1.TrainJob, err error) {
	result = &v2alpha1.TrainJob{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("trainjobs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(trainJob).
		Do(ctx).
		Into(result)
	return
}

// Update takes the representation of a trainJob and updates it. Returns the server's representation of the trainJob, and an error, if there is any.
func (c *trainJobs) Update(ctx context.Context, trainJob *v2alpha1.TrainJob, opts v1.UpdateOptions) (result *v2alpha1.TrainJob, err error) {
	result = &v2alpha1.TrainJob{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("trainjobs").
		Name(trainJob.Name).
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(trainJob).
		Do(ctx).
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *trainJobs) UpdateStatus(ctx context.Context, trainJob *v2alpha1.TrainJob, opts v1.UpdateOptions) (result *v2alpha1.TrainJob, err error) {
	result = &v2alpha1.TrainJob{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("trainjobs").
		Name(trainJob.Name).
		SubResource("status").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(trainJob).
		Do(ctx).
		Into(result)
	return
}

// Delete takes name of the trainJob and deletes it. Returns an error if one occurs.
func (c *trainJobs) Delete(ctx context.Context, name string, opts v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("trainjobs").
		Name(name).
		Body(&opts).
		Do(ctx).
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *trainJobs) DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error {
	var timeout time.Duration
	if listOpts.TimeoutSeconds != nil {
		timeout = time.Duration(*listOpts.TimeoutSeconds) * time.Second
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("trainjobs").
		VersionedParams(&listOpts, scheme.ParameterCodec).
		Timeout(timeout).
		Body(&opts).
		Do(ctx).
		Error()
}

// Patch applies the patch and returns the patched trainJob.
func (c *trainJobs) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v2alpha1.TrainJob, err error) {
	result = &v2alpha1.TrainJob{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("trainjobs").
		Name(name).
		SubResource(subresources...).
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return
}

// Apply takes the given apply declarative configuration, applies it and returns the applied trainJob.
func (c *trainJobs) Apply(ctx context.Context, trainJob *kubefloworgv2alpha1.TrainJobApplyConfiguration, opts v1.ApplyOptions) (result *v2alpha1.TrainJob, err error) {
	if trainJob == nil {
		return nil, fmt.Errorf("trainJob provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()
	data, err := json.Marshal(trainJob)
	if err != nil {
		return nil, err
	}
	name := trainJob.Name
	if name == nil {
		return nil, fmt.Errorf("trainJob.Name must be provided to Apply")
	}
	result = &v2alpha1.TrainJob{}
	err = c.client.Patch(types.ApplyPatchType).
		Namespace(c.ns).
		Resource("trainjobs").
		Name(*name).
		VersionedParams(&patchOpts, scheme.ParameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return
}

// ApplyStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating ApplyStatus().
func (c *trainJobs) ApplyStatus(ctx context.Context, trainJob *kubefloworgv2alpha1.TrainJobApplyConfiguration, opts v1.ApplyOptions) (result *v2alpha1.TrainJob, err error) {
	if trainJob == nil {
		return nil, fmt.Errorf("trainJob provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()
	data, err := json.Marshal(trainJob)
	if err != nil {
		return nil, err
	}

	name := trainJob.Name
	if name == nil {
		return nil, fmt.Errorf("trainJob.Name must be provided to Apply")
	}

	result = &v2alpha1.TrainJob{}
	err = c.client.Patch(types.ApplyPatchType).
		Namespace(c.ns).
		Resource("trainjobs").
		Name(*name).
		SubResource("status").
		VersionedParams(&patchOpts, scheme.ParameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return
}
